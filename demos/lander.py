import asyncio
from statistics import mean

import krpc
import time
import numpy as np
from krpc import services
from krpc import Client
from krpc.services.mechjeb import SmartASSAutopilotMode, SmartRCSMode
from krpc.services.spacecenter import VesselSituation
from krpc.services.ui import Text
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.constants import g
from typing import cast

from kflight import Flight
from kflight.ro import ROEngineInfo

# g = 9.81335163116455 standard gravity in KSP
MAX_STEP_IVP = 1.0  # max step size of solve_ivp in seconds (orig 2)
RPC_PORT = 50000  # ports used for the KRPC connection
STREAM_PORT = 50001
CONSOLE_OUTPUT = True  # enables text output


def vis_viva(radius, mu, sm_axis):
    """
    Returns the orbital velocity as a function of the radius `r`, the gravitational
    parameter mu, and the semi major axis of the orbit `a`.
    """
    return np.sqrt(mu * (2 / radius - 1 / sm_axis))


def delta_h(t, orbit, body, ref_frame):
    """
    Calculates the altitude difference
    between orbit and surface at time t.
    """
    pos = orbit.position_at(t, ref_frame)
    lat = body.latitude_at_position(pos, ref_frame)
    lon = body.longitude_at_position(pos, ref_frame)
    surface_h = body.surface_height(lat, lon)
    dh = body.altitude_at_position(pos, ref_frame) - surface_h
    return dh


def estimate_isp_mn_thrust(
    rated_isp: float,
    rated_thrust: float,
    isp_variance: float = 0.0001,
    mass_flow_variance: float = 0.00,
) -> tuple[float, float, float]:
    """
    Returns estimated worst-case isp, mass flow rate, and thrust
    """
    isp = rated_isp * (1 - isp_variance)
    # isp = 298.98526388889
    mass_flow = rated_thrust / (rated_isp * g) * (1 - mass_flow_variance)
    # mass_flow = 8.9430007857143
    thrust = isp * mass_flow * g
    return isp, mass_flow, thrust


def motion_equations(t, w, c, t_burn, ref_frames, sc):
    """
    Defines the 3d equations of motion of a rocket in a gravitational field
    burning retrograde in the rotating frame of the body.
    """
    # x, y, z, vx, vy, vz = w    # initial coordinates and velocity (rotated frame)
    # mu, F, m0, dm = c          # gravitational parameter and rocket-defining parameters
    # body_fixed_frame, refframe_rotating = refframes   # body reference frames

    # convert velocity to rotating frame
    vr = sc.transform_velocity(w[0:3], w[3:6], ref_frames[0], ref_frames[1])

    # normalize velocity vector
    vr = np.array(vr) / np.linalg.norm(vr)

    # transform velocity direction to fixed frame -> retrograde direction
    rg = sc.transform_direction(vr, ref_frames[1], ref_frames[0])

    # distance between vessel and center of body
    d = np.linalg.norm(w[0:3])

    m = c[2] - c[3] * (t - t_burn)
    f = [
        w[3],
        w[4],
        w[5],
        -c[1] * rg[0] / m - c[0] * w[0] / d**3,
        -c[1] * rg[1] / m - c[0] * w[1] / d**3,
        -c[1] * rg[2] / m - c[0] * w[2] / d**3,
    ]
    return f


def cost_function(t_burn, w, constants, t0, ref_frames, sc, body, orbit, end_alt: float):
    """
    Calculates the final height after hoverslam maneuver.
    """
    mu, F, m0, dm = constants
    ref_fixed, ref_rotating = ref_frames  # body reference frames

    # position at t_burn
    pos_burn = orbit.position_at(t_burn + t0, ref_fixed)

    # speed at t_burn
    speed_burn = vis_viva(orbit.radius_at(t_burn + t0), mu, orbit.semi_major_axis)

    # direction at t_burn
    dir_burn = np.array(orbit.position_at(t_burn + t0 + 0.5, ref_fixed)) - np.array(
        orbit.position_at(t_burn + t0 - 0.5, ref_fixed)
    )
    dir_burn /= np.linalg.norm(dir_burn)  # normalize
    vel_burn = tuple(speed_burn * dir_burn)  # velocity vector at t_burn

    w_burn = pos_burn + vel_burn

    t_span = (t_burn, orbit.time_to_periapsis)

    # define event functions for the optimization algorithm
    # done here to enable access to objects like sc, body and so on
    body_fixed_frame = body.non_rotating_reference_frame
    body_rotating_frame = body.reference_frame

    def finish_burn(t, y):
        """
        Calculates the current speed in the rotating frame. Terminates
        the solver when reaching 1 m/s. 0 m/s is not possible as the
        function has to change sign.
        """
        vr = sc.transform_velocity(
            y[0:3], y[3:6], body_fixed_frame, body_rotating_frame
        )
        return np.linalg.norm(vr) - 1.0

    finish_burn.terminal = True
    finish_burn.direction = -1

    # The solver evaluates EOM() within the time interval t_span. If the velocity
    # drops to zero (calculated by finish_burn) the solver stops the evaluation.
    s = solve_ivp(
        lambda t, w: motion_equations(t, w, constants, t_burn, ref_frames, sc),
        t_span,
        w_burn,
        max_step=MAX_STEP_IVP,
        events=finish_burn,
        dense_output=True,
        method="LSODA",
    )

    t_event = s.t_events[0][0]

    # position and velocity at impact or zero velocity
    f = s.sol(t_event)

    # current altitude above terrain at t_event
    lat = body.latitude_at_position(f[0:3], ref_fixed)  # latitude
    lon = body.longitude_at_position(f[0:3], ref_fixed)  # longitude
    body_rotation = t_event * body.rotational_speed * 180.0 / np.pi
    lon -= body_rotation  # compensate body rotation
    if lon < -180.0:
        lon += 360.0
    current_alt = body.altitude_at_position(f[0:3], ref_fixed) - body.surface_height(
        lat, lon
    )

    if CONSOLE_OUTPUT:
        print("%4.3f\t\t%.2f" % (t_burn, current_alt))
    return current_alt - end_alt


class SuicideBurnAP(Flight):
    info_text: Text
    perf_estimate: tuple[float, float, float]

    def update_text(self, content: str, color: tuple = (1.0, 1.0, 1.0)) -> None:
        self.info_text.text = content
        self.info_text.color = color

    def real_isp_mn_thrust(self) -> tuple[float, float, float]:
        engines = self.active_engines()

        total_thrust = 0.0
        total_mass_flow = 0.0

        # Total Isp is average of Isp weighted by mass flow
        weighted_isp = 0.0
        for engine in engines:
            info = ROEngineInfo.from_engine(engine)
            total_thrust += (info.thrust * 1000)  # info in kN
            total_mass_flow += info.mass_flow
            weighted_isp += info.isp * info.mass_flow

        total_isp = weighted_isp / total_mass_flow
        return total_isp, total_mass_flow, total_thrust

    def eval_performance(self) -> None:
        est_isp, est_mass_flow, est_thrust = self.perf_estimate
        real_isp, real_mass_flow, real_thrust = self.real_isp_mn_thrust()
        print(f"Performance evaluation:")
        isp_delta = 100.0 * (real_isp - est_isp) / est_isp
        print(f" Isp: {est_isp:.2f} s (estimated), {real_isp:.2f} s (real) | Off by {isp_delta:.2f} %")
        mass_flow_delta = 100.0 * (real_mass_flow - est_mass_flow) / est_mass_flow
        print(f" Mass flow: {est_mass_flow:.2f} kg/s (estimated), {real_mass_flow:.2f} kg/s (real) | Off by {mass_flow_delta:.2f} %")
        thrust_delta = 100.0 * (real_thrust - est_thrust) / est_thrust
        print(f" Thrust: {est_thrust:.2f} N (estimated), {real_thrust:.2f} N (real) | Off by {thrust_delta:.2f} %")

    def calculate_burn(self, end_alt: float):
        vessel = self.vessel
        conn = self.connection
        body = self.vessel.orbit.body
        orbit = self.vessel.orbit
        sc = self.sc

        est_isp, est_mass_flow, est_thrust = estimate_isp_mn_thrust(
            rated_isp=vessel.vacuum_specific_impulse,
            rated_thrust=vessel.max_vacuum_thrust,
        )
        self.perf_estimate = (est_isp, est_mass_flow, est_thrust)
        print(f"Estimated performance:")
        print(f" ISP: {est_isp:.2f} s")
        print(f" Mass flow: {est_mass_flow:.2f} kg/s")
        print(f" Thrust: {est_thrust:.2f} N")

        # collect further information about vessel and body
        body_fixed_frame = body.non_rotating_reference_frame
        body_rotating_frame = body.reference_frame
        mu = body.gravitational_parameter
        m0 = vessel.mass
        # F = vessel.max_vacuum_thrust
        F = est_thrust
        # dm = F / (vessel.vacuum_specific_impulse * g)  # mass flow (kg/s)
        dm = est_mass_flow
        # R = body.equatorial_radius

        # data streams (positions, velocities and time)
        pos = conn.add_stream(vessel.position, body_fixed_frame)
        pos_rot = conn.add_stream(vessel.position, body_rotating_frame)
        vel_fixed = conn.add_stream(vessel.velocity, body_fixed_frame)
        vel_rotating = conn.add_stream(vessel.velocity, body_rotating_frame)
        # ut = conn.add_stream(getattr, sc, "ut")  # time

        # current state used for calculations
        t0 = self.sc.ut
        pos0 = pos()
        vel0 = vel_fixed()

        # starting calculations
        self.info_text.content = "Calculating trajectory..."
        self.info_text.color = (1.0, 1.0, 1.0)

        # find the time of impact on the surface by finding the root of
        # the function delta_h using scipy.optimize.brentq
        # interval [t0, t1] in which delta_h changes its sign
        t1 = t0 + orbit.time_to_periapsis
        t_impact = cast(
            float, brentq(delta_h, args=(orbit, body, body_fixed_frame), a=t0, b=t1)
        )

        # estimate time until burn
        # the factor 0.8 is empirical
        t_burn_guess = t_impact - 0.8 * np.linalg.norm(vel_rotating()) / (F / m0)
        print(f"? Estimated burn time: {t_burn_guess - t0:.2f} s")

        # ---
        # NOTE: Times until now are expressed in KSP universal time (ut). For
        # the calculations it is easier to proceed in relative time starting
        # at t0 (see above). To shift to the relative time frame, t0 is
        # subtracted from ut.
        # ---

        # w0: initial state of the vessel
        w0 = pos0 + vel0

        # c: constants required for optimization
        c = (mu, F, m0, dm)

        # t_span: time interval in which the EOM is evaluated
        t_span_end = t_impact - t0 - 2.0

        refframes = (body_fixed_frame, body_rotating_frame)
        tt = time.time()  # measure the time of the optimization

        # find the root of cost_function and therefore the t_burn where
        # the final altitude is FINAL_ALTITUDE
        t_burn = cast(
            float,
            brentq(
                cost_function,
                0.0,
                t_span_end,
                args=(w0, c, t0, refframes, sc, body, orbit, end_alt),
                xtol=1e-3,
                rtol=1e-4,
            ),
        )

        # ---
        # evaluate the EOM again at t_burn
        # ---

        # position at t_burn
        pos_burn = orbit.position_at(t_burn + t0, body_fixed_frame)

        # speed at t_burn
        speed_burn = vis_viva(orbit.radius_at(t_burn + t0), mu, orbit.semi_major_axis)

        # direction at t_burn
        dir_burn = np.array(
            orbit.position_at(t_burn + t0 + 0.5, body_fixed_frame)
        ) - np.array(orbit.position_at(t_burn + t0 - 0.5, body_fixed_frame))
        dir_burn /= np.linalg.norm(dir_burn)  # normalize
        vel_burn = tuple(speed_burn * dir_burn)  # velocity vector at t_burn

        # -- Finish burn condition --

        # define event functions for the optimization algorithm
        # done here to enable access to objects like sc, body and so on
        body_fixed_frame = body.non_rotating_reference_frame
        body_rotating_frame = body.reference_frame

        def finish_burn(t, y):
            """
            Calculates the current speed in the rotating frame. Terminates
            the solver when reaching 1 m/s. 0 m/s is not possible as the
            function has to change sign.
            """
            vr = sc.transform_velocity(
                y[0:3], y[3:6], body_fixed_frame, body_rotating_frame
            )
            return np.linalg.norm(vr) - 1.0

        finish_burn.terminal = True
        finish_burn.direction = -1

        # -- Optimize the trajectory --

        w_burn = pos_burn + vel_burn
        t_span = (t_burn, orbit.time_to_periapsis)
        s = solve_ivp(
            lambda t, w: motion_equations(t, w, c, t_burn, refframes, sc),
            t_span,
            w_burn,
            max_step=MAX_STEP_IVP,
            events=finish_burn,
            dense_output=True,
            method="LSODA",
        )
        t_touchdown = s.t_events[0][0]

        # evaluate solution at end of burn (t_touchdown)
        final = s.sol(t_touchdown)  # pos and vel of vessel at end of maneuver
        position_final = final[0:3]
        lat = body.latitude_at_position(position_final, body_fixed_frame)
        lon = body.longitude_at_position(position_final, body_fixed_frame)
        body_rotation = t_touchdown * body.rotational_speed * 180.0 / np.pi
        lon -= body_rotation  # compensate rotation of the body
        if lon < -180.0:
            lon += 360.0
        surface_h = body.surface_height(lat, lon)
        hf = body.altitude_at_position(position_final, body_fixed_frame) - surface_h

        # final velocity
        vrf = self.sc.transform_velocity(
            position_final, final[3:6], body_fixed_frame, body_rotating_frame
        )

        # required delta_v, calculated using the rocket equation
        delta_v = est_isp * g * np.log(m0 / (m0 - dm * (t_touchdown - t_burn)))

        if CONSOLE_OUTPUT:
            print("\nOptimization finished")
            print(f"Elapsed time: {time.time() - tt:3.1f} s")
            print("\n\nResults of the optimization:")
            print(f"\nt_burn = {t_burn:4.2f} s")
            print(f"t_touchdown = {t_touchdown:4.2f} s")
            print(f"\nfinal altitude: {hf:4.1f} m")
            print(
                "final velocities: vx=%2.1f m/s, vy=%2.1f m/s, vz=%2.1f m/s"
                % (vrf[0], vrf[1], vrf[2])
            )
            print("final coordinates: (%3.4f, %3.4f)\n" % (lat, lon))
            print("required delta_v: %3.1f m/s\n" % delta_v)

        return t0, t_burn, t_touchdown, hf, vrf, delta_v

    async def landing(
        self,
        auto_warp: bool = True,
        warp_lead_time: float = 10,
        target_velocity: float = 8,
        target_end_altitude: float = 25,
        offset_burn_start: float = 0,
        stage_on_end: bool = True,
        stage_for_ullage: bool = False,
        rcs_for_ullage: bool = False,
    ) -> None:
        conn = self.connection
        sc = conn.space_center
        vessel = sc.active_vessel
        orbit = vessel.orbit
        body = orbit.body
        ref_fixed = body.non_rotating_reference_frame
        ref_rotating = body.reference_frame

        # cancel action if requirements for hoverslam are not fulfilled
        if (st := vessel.situation) not in (
            sc.VesselSituation.sub_orbital,
            sc.VesselSituation.escaping,
        ):
            self.info_text.content = f"Spacecraft status not sub-orbital: {st}"
            self.info_text.color = (1.0, 0.501, 0.078)
            return
        elif body.has_atmosphere:
            self.info_text.content = "Planet/moon has an atmosphere!"
            self.info_text.color = (1.0, 0.501, 0.078)
            return
        elif vessel.available_thrust < 1e-6:
            self.info_text.content = "No thrust available!"
            self.info_text.color = (1.0, 0.501, 0.078)
            return

        if CONSOLE_OUTPUT:
            print("\nminimizing cost function\n")
            print("t_burn \t\t altitude")

        t0, t_burn, t_touchdown, hf, vrf, delta_v = self.calculate_burn(end_alt=target_end_altitude)

        self.info_text.content = "Calculation finished"
        self.info_text.color = (1.0, 1.0, 1.0)

        # if game is paused, wait
        while conn.krpc.paused:
            time.sleep(0.2)

        # initialize krpc and mechjeb autopilots
        ap = vessel.auto_pilot
        ap.sas = False
        ap.reference_frame = ref_rotating
        mj = conn.mechjeb

        self.info_text.content = "Starting autopilot"
        self.info_text.color = (1.0, 1.0, 1.0)

        # approximate direction of velocity at t_burn
        pos1 = np.array(orbit.position_at(t0 + t_burn, ref_rotating))
        pos2 = np.array(orbit.position_at(t0 + t_burn + 2.0, ref_rotating))
        delta_pos = pos1 - pos2
        burn_vector = delta_pos / np.linalg.norm(delta_pos)

        if auto_warp:
            # Target surface retrograde
            mj.smart_ass.autopilot_mode = SmartASSAutopilotMode.surface_retrograde
            time.sleep(2)

            # rotate vessel to retrograde direction at t_burn
            # ap.engage()
            # ap.target_direction = v_burn
            # ap.wait()
            # ap.disengage()
            mj.smart_ass.autopilot_mode = SmartASSAutopilotMode.off
            ap.sas = True
            ap.sas_mode = ap.sas_mode.stability_assist

            time.sleep(0.85)

            # warp to t_burn-25s
            self.info_text.content = "Warping to burn"
            self.info_text.color = (1.0, 1.0, 1.0)
            sc.warp_to(t_burn + t0 - warp_lead_time)

        # point vessel retrograde for burn
        ap.sas = False
        mj.smart_ass.autopilot_mode = SmartASSAutopilotMode.surface_retrograde

        # Get burn engine
        eng = self.active_engines()[0]
        spool_up_time = ROEngineInfo.from_engine(eng).spool_up_time

        # wait until t_burn
        ut_burn = t0 + t_burn
        offset_ut_burn = ut_burn + offset_burn_start

        # count-down
        await self.wait_until(self.sc.ut).greater_than(offset_ut_burn - 3)
        self.ui.message("Burn starting in 3 seconds")

        # For RCS ullage, start thrusting up now
        if rcs_for_ullage:
            vessel.control.forward = 1
            while not ROEngineInfo.from_engine(eng).propellant_stability == 100:
                await asyncio.sleep(0.01)

        if stage_for_ullage:
            await self.wait_until(self.sc.ut).greater_than(offset_ut_burn - 1.5)
            vessel.control.activate_next_stage()

        # Account for spool-up time
        await self.wait_until(self.sc.ut).greater_than(offset_ut_burn - spool_up_time - 0.02)

        mj.thrust_controller.enabled = True
        mj.thrust_controller.target_throttle = 1

        # Wait until thrust
        last_ut = None
        while not ROEngineInfo.from_engine(eng).throttle == 100:
            last_ut = self.sc.ut
            await asyncio.sleep(0.005)

        # Calculate time to be mean of last_ut and ut now
        eng_on_time = self.sc.ut
        eng_on_time = mean((last_ut, eng_on_time))
        eng_on_diff = eng_on_time - ut_burn
        if eng_on_diff > 0:
            print(f"Warning: Engine-on was later than target by {eng_on_diff:.2f}s")
        else:
            print(f"Engine-on was earlier than target by {-eng_on_diff:.2f}s")

        self.info_text.content = "Suicide burn started"
        self.info_text.color = (1.0, 1.0, 1.0)

        # Evaluate estimated vs real performance
        await self.delay(1)
        self.eval_performance()

        # wait until the vessel is slowed down below target,
        # or we get a significant positive vertical velocity component
        await self.wait_until(vessel.velocity, ref_rotating).on_predicate(
            lambda v: v[1] > 5 or np.linalg.norm(v) <= target_velocity
        )
        mj.thrust_controller.target_throttle = 0
        print("\nSuicide burn finished")

        await self.delay(0.02)

        if stage_on_end:
            vessel.control.activate_next_stage()
            await self.delay(0.05)

        # Decide whether to fine tune
        # Fine-tune if altitude < 100m
        # Otherwise start another optimization
        surface_alt = self.altitude_agl()
        print(f"Burn ending alt: {surface_alt:.4f} m")

        await self.finetune_landing()

    async def finetune_landing(self, deploy_gears: bool = True, touchdown_speed: float = 1.6) -> None:
        body = self.orbit.body

        if deploy_gears:
            self.vessel.control.gear = True

        # switch to mechjeb landing autopilot
        mj = self.mechjeb
        mj.thrust_controller.enabled = False
        if CONSOLE_OUTPUT:
            print("\nSwitching to MechJeb landing autopilot")
        landing_ap = mj.landing_autopilot
        landing_ap.enabled = True
        landing_ap.touchdown_speed = touchdown_speed
        landing_ap.land_untargeted()

        # cut throttle and enable SAS when landed
        await self.wait_until(self.vessel.situation).equals(VesselSituation.landed)
        mj.thrust_controller.target_throttle = 0
        landing_ap.enabled = False

        ap = self.vessel.auto_pilot
        ap.sas = True
        ap.sas_mode = ap.sas_mode.stability_assist

        self.update_text("Landed", (0, 1, 0))
        print("landed")
        time.sleep(0.05)
        ap.sas = False

        # the actual landing coordinates
        pos = self.vessel.position(body.non_rotating_reference_frame)
        lat = body.latitude_at_position(pos, body.non_rotating_reference_frame)
        lon = body.longitude_at_position(pos, body.non_rotating_reference_frame)
        if CONSOLE_OUTPUT:
            print("\nlanding coordinates: (%3.4f, %3.4f)\n" % (lat, lon))

    async def main(self):
        # set up UI
        canvas = self.ui.stock_canvas
        screen_size = canvas.rect_transform.size
        panel = canvas.add_panel()
        panel_size = (200, 120)
        panel.rect_transform.size = panel_size
        panel.rect_transform.position = (
            screen_size[0] // 2 - panel_size[0] // 2 - 50,
            200,
        )
        title_panel = panel.add_panel()
        title_panel.rect_transform.size = (panel_size[0], 24)
        title_panel.rect_transform.position = (0, panel_size[1] // 2 - 12)
        title = title_panel.add_text("Suicide Burn AP")
        title.alignment = self.ui.TextAnchor.middle_center
        title.color = (1.0, 1.0, 1.0)
        title.size = 13

        # add button to start procedure
        button = panel.add_button("Start descent guidance")
        button.rect_transform.size = (panel_size[0] - 20, 30)
        button.rect_transform.position = (0, 12)

        # add text output
        info_text = panel.add_text("")
        info_text.rect_transform.position = (0, -40)
        info_text.rect_transform.size = (panel_size[0] - 20, 60)
        info_text.alignment = self.ui.TextAnchor.upper_left
        info_text.color = (1.0, 1.0, 1.0)
        info_text.size = 13
        self.info_text = info_text

        await self.wait_until(button.clicked)
        button.clicked = False
        await self.landing(
            target_end_altitude=20,
            offset_burn_start=-0.3,
            # stage_for_ullage=False,
            rcs_for_ullage=True,
            stage_on_end=True
        )
        self.close()


if __name__ == "__main__":
    SuicideBurnAP().run()

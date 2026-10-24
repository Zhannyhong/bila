# `insert_book.py` Pseudocode

Small context: this file runs the robot-side insertion behavior for clear, tight, and packed shelf rows.

## High-level sequence

The file follows this overall logic:

- build a scene that includes the target book,
- reason about all geometry in the shelf frame,
- choose safe arm configurations,
- move the hand using inverse kinematics,
- insert the book,
- widen the gap first if needed,
- score the result.

## Constants

Define many constants for:

- Panda hand geometry,
- thin-finger geometry,
- grasp depth and grasp height,
- insertion overlap and hover height,
- safety clearances,
- force thresholds that mean “stop pushing”,
- move speeds,
- gap-widening margins,
- packed-row opening behavior,
- success tolerances.

These constants tell the scripted controller how aggressively or conservatively to act.

## Scene setup

Inside `build_model(cfg, row, panda_xml)`:

- Build the base scene XML from `shelf_scene.build_scene`.
- Add the target book into the world at a parked position away from the shelf.
- Extend the keyframe so the parked target book has an initial pose.
- Before compiling:
  - enable gravity compensation on Panda bodies.
- Compile the MuJoCo model.
- Strengthen the gripper actuator so it can actually hold books.
- Return the model.

## Shelf-frame helper

Define `ShelfFrame` to store:

- the shelf rotation matrix,
- the shelf origin in world coordinates.

Inside `ShelfFrame.rot(yaw)`:

- Return a world rotation matrix for something yawed inside the shelf frame.

Inside `ShelfFrame.to_world(p)`:

- Convert shelf-frame coordinates into world coordinates.

Inside `ShelfFrame.to_shelf(p)`:

- Convert world coordinates into shelf-frame coordinates.

Inside `ShelfFrame.box(model, data, geom)`:

- Get all eight box corners of a geom.
- Transform them into the shelf frame.
- Return those corners.

## Collision geometry helpers

Inside `obstacles(model, data, shelf, row)`:

- For every row book and each side panel:
  - get its box corners in the shelf frame,
  - compute its left x, right x, and front y.
- Return the obstacle list.

Inside `tip_limit(obs, x_mid, parts)`:

- For each relevant part of the hand:
  - check which obstacles overlap in x,
  - compute how far forward the fingertips may go without that part colliding.
- Return the smallest safe fingertip y limit.

## Pose math helpers

Inside `mat2quat(R)`:

- Convert a rotation matrix into a quaternion.

Inside `quat2mat(q)`:

- Convert a quaternion into a rotation matrix.

Inside `rotation_between(R0, R1)`:

- Compute the rotation vector that takes `R0` to `R1`.

Inside `interpolate(p0, R0, p1, R1, s)`:

- Move the position in a straight line from start to end.
- Interpolate the orientation smoothly.
- Return the intermediate pose.

## Inverse kinematics

Define `ArmIK`.

Inside `ArmIK.__init__(model, q_rest)`:

- Create a private MuJoCo data object for planning.
- Find the Panda hand body.
- Find the joint position and velocity addresses for the seven arm joints.
- Store joint limits.
- Store the preferred rest posture.
- Create Jacobian buffers.
- Record which geoms belong to the robot.
- Record which geoms belong to the fingers.
- Record the finger joint addresses.
- Start with an empty ignore set.

Inside `sync(data, finger)`:

- Copy the current simulation state into the IK planner.
- If a finger opening override is given:
  - set both finger joints to that opening.

Inside `collides(q, allowed)`:

- Put the arm at the candidate joint state.
- Run forward kinematics.
- For each contact:
  - ignore robot-robot contacts,
  - ignore shallow contacts,
  - allow finger contact with explicitly allowed objects,
  - otherwise report collision.
- Return whether a disallowed collision exists.

Inside `fk(q)`:

- Put the arm at the candidate joint state.
- Run kinematics.
- Return the hand position and hand rotation.

Inside `error(q, pos, rot)`:

- Compute the current hand pose.
- Return the combined position and orientation error to the target pose.

Inside `solve(q, pos, rot, iters, max_step)`:

- Repeat for a few iterations:
  - compute pose error,
  - compute the hand Jacobian,
  - repeatedly remove joints that are pinned at limits and being pushed further into the limit,
  - solve a damped least-squares IK update,
  - add a null-space pull toward the rest posture,
  - clip the step size,
  - clip to joint limits.
- Return the updated joint state.

Inside `margin(q)`:

- Return how far the closest joint is from its limit.

Inside `solve_far(q, pos, rot)`:

- Use the provided seed as the preferred rest posture.
- Run many IK iterations.
- Return the final joint state and whether the pose converged.

Inside `follow(q, poses, floor, allowed)`:

- Track a list of desired hand poses one by one.
- Re-solve IK at each step.
- Track the worst joint-limit margin on the path.
- Stop early if:
  - the margin falls below the requested floor, or
  - a forbidden collision appears.
- At the end:
  - check whether the final pose error is small enough.
- Return the end joint state, the worst margin, and whether the path succeeded.

## Motion runner

Define `Runner`.

Inside `Runner.__init__(...)`:

- Store the model, data, IK solver, current commanded joints, and grip setting.
- Read the current hand pose from forward kinematics.
- Store viewer and logging handles.
- Record timing information.
- Find the wrist force sensor index.

Inside `wrist_force()`:

- Return the wrist force vector.

Inside `halt()`:

- Freeze the commanded arm state at the actual current arm state.
- Update the stored hand pose.

Inside property `t`:

- Return elapsed simulated time since the runner started.

Inside `step()`:

- Send the commanded arm joints and gripper command.
- Step the MuJoCo simulation.
- If a viewer exists:
  - draw at about 60 frames per second,
  - keep rendering close to real time.

Inside `move(name, pos, rot, speed, grip, stop_force)`:

- Log the move name.
- If a grip command was provided:
  - update the gripper target.
- If no target rotation was provided:
  - keep the current rotation.
- Measure distance and rotation to the target.
- Convert that into a min-jerk move duration.
- Smooth the wrist force along the hand approach axis.
- For each interpolation step:
  - compute the next desired pose,
  - solve IK for that pose,
  - step the simulation,
  - update the smoothed force.
- If a stop-force threshold was provided and the smoothed force changes too much:
  - halt,
  - log the stop,
  - return failure.
- If the move reaches the end:
  - return success.

Inside `move_joints(name, q_goal, speed)`:

- Log the move.
- Interpolate directly in joint space with a min-jerk profile.
- Step the simulation along the path.
- Update the stored hand pose at the end.

Inside `wait(name, duration, grip)`:

- Log the wait.
- If a grip command was provided:
  - update the gripper target.
- Step the simulation for the requested duration.

## Small helper functions

Define:

- `hand_pose_for_point`
- `segment`

These exist to:

- place a chosen point on the hand at a chosen world point,
- build straight-line pose segments.

## Episode object

Define `Episode`.

Inside `Episode.__init__(model, data, row, viewer, log)`:

- Store the model, data, row, viewer, and logger.
- Create a `ShelfFrame`.
- Store the shelf axes in world coordinates.
- Read target-book dimensions.
- Read shelf dimensions and target-board height.
- Check that the shelf level is tall enough to rotate the hand for the final push.
- Record the target book geom, body, and joint ids.
- Record the row book geoms and bodies.
- Save the starting positions and starting tilt angles of the row books.
- Compute the seated spine depth target.
- Compute where the pads contact the book in the book frame.
- Compute the hand height used for pushing.
- Compute the resting book-center height on the board.
- Create an `ArmIK`.
- Tell the IK solver to ignore collisions with the target book.
- Start with no runner.
- Start with an empty info dictionary.

## Episode geometry methods

Inside `obstacles()`:

- Return the current shelf obstacles.

Inside `slot()`:

- Measure the current left and right neighbor box corners.
- Extract the corners that lie on each spine.
- Return:
  - left face x,
  - right face x,
  - left spine y,
  - right spine y.

Inside `free_space()`:

- Measure the slot.
- Compute how much row space exists to the left and right once existing book thickness is accounted for.
- Return left and right free-space values.

Inside `seat_target()`:

- Measure current spine depths of the neighboring books.
- If the target book can finish flush with the more protruding neighbor before hitting the back panel:
  - return that flush target and the rule `flush`.
- Otherwise:
  - return a target that places the book sufficiently inside the shelf front edge and the rule `inside`.

Inside `book_pre(x_slot)`:

- Return the target book center pose at the pre-insertion position.

Inside `hold_limit(...)` and `push_limit(...)`:

- Use `tip_limit` to compute how deep the hand can go while holding or pushing.

## Episode hand-pose methods

Inside `grasp_pose(sign, centre, yaw)`:

- Build the desired book rotation in the shelf frame.
- Build the hand rotation that grasps the book from the correct side.
- Compute the hand position that places the gripping pads correctly on the book.
- Return the hand pose.

Inside `push_pose(sign, x, tip_y)`:

- Rotate the hand so the fingers are stacked vertically.
- Place the lower fingertip at the requested shelf-frame point.
- Return that pushing hand pose.

Inside `col_pose(sign, x, tip_y, z_axis)`:

- Build a vertically stacked-finger pose at a chosen fingertip location and hand-axis height.

Inside `tip_y(pose)`:

- Compute the fingertip y position of the provided hand pose.

Inside `push_path(start, sign, x, spine_now, tip_goal)`:

- Choose a safe back-off depth behind the current book spine.
- Build:
  - a back-off pose,
  - a ready-to-push pose,
  - a whole path that backs off, rotates, and then pushes forward.
- Return all three.

Inside `insertion_plan(...)`:

- Estimate where the hand can keep holding the book while sliding it in.
- Build the nominal slide path.
- Build the nominal push path after release.
- Return the full planned pose sequence.

## Choosing arm configurations

Inside `choose_arm(candidates, allowed, finger)`:

- Sync the IK scene with the current simulation.
- For several rounds:
  - generate IK seeds,
  - include the Panda home pose in the first round,
  - for each candidate start pose and path:
    - solve IK from each seed,
    - if the start pose is reachable:
      - follow the whole path,
      - reject paths that collide or go too near joint limits,
      - keep the candidate with the best worst-case joint-limit margin.
- If no safe candidate exists:
  - raise an error.
- Set the IK rest posture to the chosen arm configuration.
- Return the chosen joint state and its label.

## Starting and repositioning the arm

Inside `place_arm(q)`:

- Put the arm at the given joint state.
- Open the fingers.
- Zero velocities.
- Run forward dynamics.
- Create a new `Runner`.

Inside `begin(candidates, allowed, finger)`:

- If a runner already exists:
  - try to continue from the current hand pose directly into one of the new candidate paths.
  - If one candidate is reachable with enough joint-limit margin:
    - move to that candidate’s start pose,
    - return its label.
- Otherwise:
  - choose a fresh arm configuration using `choose_arm`.
  - If the runner does not exist yet:
    - place the arm there.
  - Otherwise:
    - reposition the arm there.
- Return the chosen label.

Inside `reposition(q)`:

- First pull the hand straight out of the shelf if needed.
- Check whether the arm can move directly in joint space to the new configuration without collision.
- If not:
  - check whether it can go via the tucked Panda home pose.
- If a safe path exists:
  - execute the joint-space move.
- Otherwise:
  - raise an error.

## Taking and tracking the target book

Inside `take_book(grasp_sign, centre, yaw, name)`:

- Put the target book physically between the open fingers.
- Set the finger joints so they are just wider than the book.
- Place the target book so it matches the intended grasp pose.
- Zero the target book velocities.
- Run forward dynamics.
- Close the gripper and wait.
- After closing:
  - measure the true relative pose of the book inside the hand.
- Save that relative pose so later hand motions carry the book consistently.

Inside `hand_for_book(centre, yaw)`:

- Combine the desired book pose with the measured in-hand transform.
- Return the hand pose needed to place the book there.

## Seating the book

Inside `seat(x_slot, y_from, push_sign)`:

- Measure current obstacles.
- Compute the current final seating target.
- Compute how far inward the hand can still hold the book.
- If the hand cannot get the book far enough onto the board to safely let go:
  - raise an error.

- Move to rest the book on the board.

- Slide the book inward along the board.
- If the wrist force says the book jammed:
  - stop the slide early.
- If the book stopped too early and is still not properly over the board:
  - raise an error.

- Open the gripper and record the release time.

- Re-measure the row because the slide may have moved the neighbors.
- If the slide already seated the book:
  - back off,
  - let the scene settle,
  - return release time, completion time, and whether the hand was blocked.

- Otherwise:
  - re-measure the current book position,
  - choose the final push target,
  - decide whether nearby spines will block the hand short of the true goal.

- Try the preferred push-hand orientation first.
- If that orientation would drive the arm near a joint limit:
  - try the opposite orientation.
- If neither is safe:
  - raise an error.

- Back off.
- Rotate into pushing orientation.
- Reach the book spine.
- Push the book toward the target, using the wrist-force threshold to stop if it jams.

- Repeat a few times:
  - re-measure the neighbors,
  - if the target moved deeper because neighbors shifted:
    - push further.
  - stop if the book is already in place, jammed, or the hand cannot go deeper.

- Record the completion time.
- Back the hand out.
- Let the scene settle.
- Return release time, completion time, and whether the hand was blocked.

## Scoring

Inside `score(t_start, t_release, t_done, blocked, **info)`:

- Measure the final target-book box.
- Measure the final target-book tilt.
- Measure how far the book sits from the back panel.
- Measure where the spine ended up.
- Recompute the current target seating rule.

- If the seating rule is `flush`:
  - compute offset from the more protruding neighbor spine,
  - declare in-place success if the offset is within tolerance.

- If the seating rule is `inside`:
  - compute how far inside the front edge the spine ended up,
  - declare in-place success if it is deep enough.

- Check whether the robot is still touching the book.
- Measure how much every row book moved.
- Measure how much every row book changed tilt.

- Build a result dictionary containing:
  - upright status,
  - in-place status,
  - released status,
  - rule used,
  - offsets,
  - tilt,
  - back gap,
  - blocked status,
  - neighbor disturbance,
  - timing,
  - any extra info.

- Define overall success as:
  - upright,
  - in place,
  - released.

- Return the result dictionary.

## Clear-scenario insertion

Inside `insert(method="none needed")`:

- Measure the current slot.
- Compute the slot center.
- Compute the pre-insertion book pose.
- Measure current obstacles.

- Build candidate plans for:
  - both possible grasp signs,
  - both possible push signs.

- Begin from the best candidate.
- If this is the first step of the episode:
  - take the book already at the pre-insertion pose.
- Otherwise:
  - take the book through an unmodelled hand-off.

- Record the start time.
- Run `seat`.
- Score the result.
- Return the score.

## Tight-scenario insertion

Inside `insert_tight(strategy, margin)`:

- Measure the current slot and gap width.
- Compute how much additional widening is needed.
- Record that widening amount.

- If no widening is needed:
  - log that the gap already fits.
  - use ordinary insertion.

- Otherwise:
  - plan a fingertip widening strategy.
- If planning fails:
  - skip the episode with the stored reason.

- Record whether the method is one-sided or both-sided fingertip widening.
- Run `fingertip_then_insert`.

Inside `plan_fingertip(...)`:

- If the fingers do not physically fit the gap:
  - store the reason and fail.

- Measure free row space on both sides.
- Measure current obstacles.
- Compute a split-widening candidate.
- Compute one-sided widening candidates ordered by which side has more spare room.

- Depending on the chosen strategy:
  - try split first or one-sided first.

- For each candidate widening plan:
  - reject impossible or negative moves,
  - compute how deep the fingertips can reach,
  - reject plans where the fingertips cannot reach behind the necessary neighbor spine.

- Return the first valid plan.
- If no valid plan exists:
  - store the reason and fail.

Inside `fingertip_depth(xs, obs)`:

- Compute how deep the fingertips may go while the palm stays clear over the whole sideways sweep.

Inside `fingertip_then_insert(...)`:

- Compute key fingertip positions for entering, pushing, crossing, and exiting.
- Build candidate whole-arm plans that include:
  - putting the fingers into the gap,
  - pushing neighbors,
  - returning to the slot center,
  - taking the fingers out,
  - moving to the pre-insertion grasp pose,
  - performing the insertion plan.

- Choose the best candidate.
- Open the gripper if needed.
- Move the fingers into the gap.

- If left widening is needed:
  - push the left neighbor.
- If right widening is needed:
  - push the right neighbor.

- If only one side was pushed:
  - check whether the other side must be topped up.
  - if so:
    - push the other side too.

- Return to the middle.
- Take the fingers out.
- Record the gap-widening time.

- Re-measure the slot.
- If the new gap is still narrower than the book:
  - skip the attempt.

- Compute the new slot center.
- Move to the pre-insertion grasp pose.
- Take the book.
- Run the normal `seat` routine.
- Score and return the result, including widening details.

Inside `top_up(...)`:

- Measure the new slot after the first push.
- Compute how much gap is still missing.
- If the remaining shortage is tiny:
  - do nothing.

- Compute where the hand would have to move to push the other side.
- Check whether the fingertips can reach that neighbor.
- Check whether the arm can safely complete the rest of the insertion after making that push.
- If not:
  - log why and do nothing.

- Otherwise:
  - perform the extra push.
- Return the updated slot information and distance pushed.

Inside `push_neighbour(push_sign, side, target, limit, tip_in, tries)`:

- Repeatedly command the fingertip push toward the target mark.
- After each push:
  - measure the actual new slot face,
  - compute how much shortfall remains.
- If there is still meaningful shortfall:
  - aim beyond the mark by the missing amount.
- Stop when the target is close enough or the side limit is reached.

## Packed-scenario insertion

Inside `insert_packed(margin)`:

- Measure how much widening is needed.
- Record that widening amount.

- If no widening is needed:
  - use ordinary insertion.

- Measure free row space on each side.
- If there is not enough free space in the row:
  - skip the attempt.

- Choose the better side for opening.
- Run `stagger_open` to create room for the fingers.
- If opening fails:
  - skip the attempt.

- After opening:
  - run the tight-gap insertion flow.
- Update the reported method string so it includes the stagger-open stage.
- Return the result.

Inside `stagger_open(side, opening)`:

- Measure the current slot.
- Choose the book face that will be pushed aside and the other book that may need to be pushed backward first.
- Compute how much room exists behind that other book.
- If there is no room to push it back enough:
  - store the reason and fail.

- If that other book is thinner than the fingers:
  - store the reason and fail.

- Compute:
  - where the fingers should contact,
  - how far back to push,
  - how far aside to push,
  - how far out the hand should start.

- Build candidate hand paths for both hand orientations.
- Choose a safe arm path.
- If no safe path exists:
  - store the reason and fail.

- Close the gripper.
- If a backward push is needed:
  - reach the other neighbor spine,
  - push that neighbor back into the shelf.

- Put the fingertips beside the newly exposed face.
- Push that face sideways.
- If the arm gives way and the slot is still too small:
  - push further.

- Ease the fingers off the face before withdrawing them.
- Take the fingers out.
- Let the row settle.
- Return success.

## Skip helper

Inside `skip(reason)`:

- Log that this attempt was skipped.
- Measure how far the row books moved.
- Return a failure dictionary containing the reason and the disturbance metrics.

## Episode and CLI helpers

Inside `make_episode(cfg, scenario, rng, panda_xml, max_tries)`:

- Repeatedly sample a row.
- Build the full insertion model.
- Create MuJoCo data.
- Settle the scene.
- If the row is stable:
  - return model, data, row.
- If repeated attempts fail:
  - raise an error.

Inside `report(res)`:

- If the attempt was skipped:
  - format a skipped message.
- Otherwise:
  - build a human-readable success or failure summary using:
    - target tilt,
    - placement offset,
    - whether the hand was blocked,
    - whether the book was released,
    - how much other books moved,
    - how long the attempt took.

## Command-line entry point

Define `main`.

- Create a command-line parser.
- Add arguments for:
  - scenario,
  - seed,
  - number of trials,
  - tight-gap strategy,
  - gap margin,
  - Panda XML path,
  - viewer mode.
- Parse the arguments.

- Define `attempt(ep)`:
  - if the scenario is clear:
    - run ordinary insertion,
  - if the scenario is packed:
    - run packed insertion,
  - otherwise:
    - run tight insertion.

- For each requested seed:
  - build one insertion episode,
  - print a summary of the slot and target book,
  - choose a verbose logger for single-trial mode and a quiet logger for batch mode.

- If viewer mode is enabled:
  - open the MuJoCo viewer,
  - apply the default camera,
  - run the episode,
  - print the result,
  - keep the viewer alive until the window closes.

- Otherwise:
  - run the episode headlessly.
  - If a runtime error happens:
    - record it as a failure result and print the error.

- Store every result.

- If multiple trials were requested:
  - print total successes,
  - print counts of blocked, non-upright, and unreleased failures,
  - print grouped error counts,
  - print grouped skipped reasons,
  - print grouped method statistics.

If this file is being run directly:

- Call `main`

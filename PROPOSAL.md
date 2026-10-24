# Perception-Conditioned Gap Creation and Residual Reinforcement Learning for Adaptive Robotic Book Reshelving

## Abstract

This proposal studies robotic book reshelving as a closed-loop manipulation problem in which the robot repeatedly observes the current shelf state, chooses actions that modify that state, and aims to reach a desired final state where the target book is correctly placed and the neighboring row remains stable. The main challenge is that realistic shelf rows are not uniformly accessible: some present a clear insertion gap, while others are tight, leaning, or collapsed and therefore require row adjustment before insertion can succeed. The immediate research question is how to couple RGB-D perception with robot action so that the system can adapt across these conditions. The proposed near-term approach is a structured closed-loop controller with three roles: state estimation, action selection, and contact-robust execution. This structure is not the final claim about the best architecture; rather, it is a practical way to make the full task testable, interpretable, and data-generating under current hardware constraints. The longer-term research direction is to investigate whether the same state-to-action mapping can be learned by a more integrated vision-language-action model trained from multimodal observations and closed-loop trajectories. The expected outcome is a clearer understanding of which shelf states require row adjustment, when residual learning improves end-to-end reshelving, and how to bridge from a structured controller to a larger integrated policy.

## 1. Motivation

Robots are increasingly used in libraries for shelf scanning and inventory, yet reshelving remains difficult because successful placement depends on both understanding the local shelf configuration and executing contact-rich manipulation under uncertainty. In realistic bookshelves, the target slot is often not neatly open. Neighboring books may lean into the space, compress the row, or collapse into the available gap, making direct insertion impossible or unreliable.

Prior work addresses only part of this challenge. Sygo et al. demonstrated an integrated library reshelving pipeline on a bimanual PR2 robot using general-purpose manipulators, combining book spine detection, catalog-based identification, collision-aware motion planning, and a scripted force-controlled placement routine. Their work shows that an end-to-end reshelving system is feasible, but it assumes books are already reasonably upright and places books against a shelf wall or a row boundary rather than between closely packed neighbors. The reported failures also suggest that scripted contact handling is brittle when the environment model is inaccurate or book covers are compliant.

Liu et al. (2026) addressed the final insertion phase more directly by combining nominal task-space control with residual PPO. In that formulation, the nominal controller handles the structured insertion motion, while the learned residual provides bounded local corrections and decides when to release the object. This improves robustness to contact uncertainty, but it assumes the target slot is already known, accessible, and reached by a higher-level planner. As the authors note, residual learning can compensate for moderate modeling error, but it cannot create physical clearance when the geometry does not permit insertion.

These works leave a central research gap: realistic reshelving requires the robot to jointly decide **what the shelf state is**, **whether direct insertion is feasible**, and **which manipulation strategy should be applied**. This project studies that missing coupling between perception and robot action.

## 2. Research Goal

The goal of this project is to study robotic book reshelving as a **closed-loop state-action-state problem**:

- **perception** estimates the current shelf state,
- **action** selects and executes robot movements that change that state, and
- **success** is defined by reaching the desired final state in which the target book is correctly seated in its slot and the row remains stable.

Under this framing, the research focus is the **entire reshelving process**, not a comparison between isolated components. The key question is how the robot should map observed shelf state to action so that it can adapt across clear gaps, tight rows, leaning neighbors, and collapsed rows.

The project uses modular components such as perception, gap creation, and insertion not because they are the final answer, but because they provide a practical and testable scaffold for studying the full loop with current hardware and data constraints.

## 3. Problem Statement

Given a grasped book and its correct target position in an ordered row of shelf books, the robot must:

1. identify the correct insertion slot using the spine label of the grasped book and the surrounding shelf books;
2. perceive the local row state from RGB-D observations, including neighbor poses, lean angles, visible free space, and uncertainty in these estimates;
3. decide whether direct insertion is possible or whether the row must first be rearranged;
4. create or widen a gap by displacing neighboring books along the shelf axis when necessary; and
5. insert and seat the target book without damaging it, toppling neighbors, or disturbing the intended row order.

The main scientific challenge is therefore **adaptive manipulation under perceptual and contact uncertainty**, rather than insertion control alone.

## 4. Core Research Question

How can a robot use perception of the current shelf state to choose and execute actions that transform the row into the desired final state, namely a correctly placed and seated book, across different scenarios such as clear gaps, tight gaps, leaning neighbors, and collapsed rows?

## 5. Research Sub-Questions

1. Can the robot infer enough about the current row state from RGB-D observations to choose actions that reliably progress toward a successful final shelf state?
2. Does a closed-loop policy that can alternate between row adjustment and insertion outperform an insertion-only policy when the shelf is tight, leaning, or collapsed?
3. Does explicit row adjustment increase the set of initial shelf states from which successful reshelving is possible?
4. Does residual learning improve the robustness of the full reshelving loop once the robot must deal with imperfect perception and imperfectly created gaps?
5. Can an integrated policy representation, such as a VLA-style model, learn useful state-to-action mappings from multimodal observations, and how does it compare with a structured modular controller under the same scenarios?

## 6. Why a Structured Approach Still Helps

The task structure suggests that a single residual PPO policy is unlikely to be the most sensible **starting point**. The difficulty does not come only from local contact dynamics. It also comes from the fact that different scenarios have different **geometric feasibility conditions**:

- in a clear-gap case, the book may be inserted directly;
- in a tight-gap case, small contact corrections may suffice;
- in a leaning-neighbor case, the robot may need to straighten or push books before insertion;
- in a collapsed-row case, insertion may be impossible without first creating space.

Because some scenarios require **changing the geometry of the row** before insertion, a structured perception-conditioned system is a useful first research vehicle:

- a **state estimator** represents the current row condition;
- an **action policy** chooses whether to adjust the row, insert, or continue correcting; and
- a **controller** executes those actions robustly under contact uncertainty.

This decomposition should be viewed as an **instrumented approximation of the full loop**, not as a claim that the final system must remain modular. It allows the project to test the overall state-to-action-to-goal process with the current setup, while also generating data and insight for later integration into a larger end-to-end model.

## 7. Methodology

### 7.1 Formal Task Formulation

The task is formulated as a partially observed sequential decision problem. At each time step `t`, the robot receives an observation `o_t` of the current shelf and chooses an action `a_t` that changes the shelf configuration and robot state, producing a new observation `o_(t+1)`.

The observation should include:

- RGB-D views of the local shelf region;
- the identity of the target book and the target slot implied by shelf order;
- estimated neighbor boundaries, lean directions, visible free space, and contact-relevant geometry;
- recent action history and end-effector state; and
- optional force or tactile feedback if available.

The latent environment state `s_t` contains the physical shelf configuration that the robot only partially observes, including true book poses, frictional contacts, compliance, and unobserved occluded geometry.

The action `a_t` should be defined at the robot level rather than the perception-module level. A practical action space for the current project is:

- approach or reposition the book relative to the slot;
- push or slide neighboring books along the row axis;
- perform insertion motion with corrective adjustments;
- release, reseat, or retreat.

Under this formulation, success is not a correct intermediate prediction but a successful transition to a goal state `s_goal` in which:

- the target book occupies the correct shelf position,
- the book is fully seated,
- neighboring books remain within acceptable displacement bounds, and
- the row remains stable and undamaged.

### 7.2 Closed-Loop Controller For The Current Project

The current project realizes this task through a structured closed-loop controller with three coupled roles.

#### A. State Estimation

An RGB-D sensing pipeline estimates the local row condition:

- visible book identities and order from spine labels;
- the target insertion interval;
- neighboring book boundaries and approximate poses;
- lean angles, clutter, and row disorder indicators;
- visible depth and estimated clearance; and
- uncertainty in the state estimate.

The purpose of this module is not to solve perception in isolation. Its purpose is to provide a usable task state for choosing the next robot action.

#### B. Action Selection

Given the current estimated state, the robot chooses the next behavior that is most likely to move the shelf toward the goal state. For the initial system, this can be represented as a small action set:

- **insert** when the estimated slot is accessible;
- **adjust-row** when the target interval is blocked or too narrow;
- **correct-and-retry** when the previous attempt produced partial progress;
- **terminate** when success or an unrecoverable failure condition is detected.

This action selection can initially be rule-based or threshold-based and later replaced by a learned policy. The key point is that the decision is made in closed loop from the current state estimate.

#### C. Contact-Robust Execution

Once an action is selected, the robot executes it using nominal control augmented with local learning where helpful:

- for **insert**, the nominal controller performs insertion and seating, while residual PPO provides bounded corrective actions under contact uncertainty;
- for **adjust-row**, the nominal controller performs structured pushing or sliding of neighboring books, with optional learned refinement for difficult contacts;
- for **correct-and-retry**, the controller updates pose, compliance, or contact approach based on the observed result of the previous step.

Residual RL is therefore used as a **local executor inside a larger closed-loop policy**, rather than as the sole decision-maker for the entire task.

### 7.3 Why This Still Tests The Full Process

Although the controller is described in modules, the experiment remains end-to-end because the evaluation target is always the same: does the robot transform the initial shelf state into the desired final shelf state?

The modular structure is used for three practical reasons:

1. it makes the overall process implementable with current hardware and data;
2. it makes failure modes observable and diagnosable; and
3. it produces trajectories that can later supervise a more integrated policy.

### 7.4 Simulation Definition And Adjustable Parameters

The current simulation is defined by explicit parameter ranges in the scene generator rather than by a single fixed shelf. This is intentional: the benchmark should cover realistic variation while still staying within a controllable and measurable task family.

The main simulation choices are as follows.

- **Shelf width**: `0.60-1.00 m`. This allows rows long enough to contain multiple neighbors and nontrivial free space while remaining consistent with a single robotic workstation.
- **Shelf depth**: `0.22-0.32 m`. This covers common bookshelf-like depths and ensures that insertion is meaningfully front-to-back constrained.
- **Number of levels**: `3-5`. This creates full-bookcase scenes rather than a single isolated cubby, while still keeping the task computationally manageable.
- **Target level height**: sampled from level heights `0.25-0.40 m` with the target board top constrained to `0.40-1.10 m`. These values are chosen so the Panda can reach the row while keeping the hand roughly level with the shelf during insertion.
- **Book thickness**: `0.015-0.06 m`. This spans thin paperbacks to thicker hardcovers and directly changes how difficult it is to create or exploit a gap.
- **Book depth**: `0.12-0.22 m`, with every book at least `0.03 m` shallower than the shelf. This prevents unrealistic back-panel penetration while still allowing meaningful variation in fore-aft contact.
- **Book height**: `0.17-0.30 m`. This covers a realistic range of library-sized books and affects both tipping stability and the vertical shape of the free gap.
- **Book density**: `400-900 kg/m^3`. This creates variation in mass from lighter softcover-like objects to heavier hardback-like objects without making the objects unrealistically heavy.
- **Shelf-board friction**: `0.3-0.8`. This range changes how easily books slide during widening and insertion and is intended to cover smoother and grippier shelf materials.
- **Book friction**: `0.25-0.7`. This changes inter-book contact behavior and how strongly neighbors resist being pushed aside.
- **Book spacing**: `0.0-0.003 m`. This allows nearly touching rows as well as lightly separated rows without introducing obviously artificial gaps.
- **End slack**: `0.03-0.15 m`. This controls how much extra row space exists at one end and therefore how much sideward redistribution is physically possible.
- **Contact solver time constant**: `0.005-0.02`. This varies contact stiffness across episodes and helps test robustness to different contact responses.
- **Shelf pose jitter**: up to `0.01 m` in planar position and `2 deg` in yaw. This prevents the benchmark from becoming too tied to a single canonical shelf placement.

The scenario-specific parameters are also explicit and adjustable.

- **Clear-gap factor**: `1.2-2.0` times the target book thickness. This defines easy cases where insertion should be feasible without row adjustment.
- **Tight-gap factor**: `0.6-1.1` times the target book thickness. This defines rows where insertion may require adaptation or widening.
- **Leaning angle**: `12-35 deg`. This range is large enough to create real obstruction of the slot while remaining physically plausible for books that lean and settle against their neighbors.
- **Leaning reserve**: `0.18 m`. This caps how much row space is reserved for a leaning obstruction.
- **Collapsed reserve**: `0.15 m`. This caps the hole that a collapsed run falls into.
- **Collapsed run length**: `3-6` books. This creates meaningful multi-book disorder without making the task too rare or excessively chaotic.

The robot embodiment is also part of the simulation definition.

- **Robot**: Franka Panda mounted in front of the shelf, with base height coupled to the target shelf level so the hand sees a similar insertion geometry across levels.
- **Fingers**: custom thin fingers of width `10 mm` across their travel. These are chosen because the standard Menagerie fingers are too wide for many generated tight gaps, which would make the benchmark artificially impossible.
- **Wrist sensing**: a wrist force/torque sensor is included so contact-sensitive stopping and future force-conditioned policies are possible.
- **Wrist camera**: a wrist RGB-D-like viewpoint is assumed in the perception design, while a fixed scene camera supports broader observation.

These values are not meant to be permanent constants. They are chosen as a starting benchmark regime for three reasons:

1. they are broad enough to create meaningful diversity in row geometry and contact behavior;
2. they remain within plausible ranges for book shelving rather than arbitrary box insertion; and
3. they stay within the reach and force limits of the current Panda-based simulation setup.

All of these values can be adjusted in the generator. This is important scientifically because the benchmark should support controlled ablations.

For example, the project can later narrow or widen:

- book thickness ranges to test size generalization,
- friction ranges to test contact robustness,
- leaning angles to test perception-action adaptation under increasing obstruction,
- gap factors to control how often widening is required,
- shelf dimensions to study transfer across furniture geometries.

The proposal therefore treats the current parameter ranges as a **benchmark definition**, not as a fixed claim about the only valid simulation setting. The ranges are chosen to be realistic, interpretable, and adjustable.

## 8. Illustration: Perception-to-Action Pipeline

```text
Figure 1. Proposed adaptive reshelving pipeline

   RGB-D shelf view + spine labels
                 |
                 v
      +------------------------+
      | Row-state estimation   |
      | - slot location        |
      | - free space           |
      | - neighbor poses       |
      | - lean / disorder      |
      | - uncertainty          |
      +------------------------+
                 |
                 v
      +------------------------+
      | Strategy selection     |
      | Direct insert?         |
      | Or create gap first?   |
      +------------------------+
           |               |
           |               |
           v               v
 +----------------+   +-------------------+
 | Direct insert  |   | Gap creation      |
 | nominal ctrl + |   | nominal ctrl +    |
 | residual PPO   |   | optional residual |
 +----------------+   +-------------------+
           \               /
            \             /
             v           v
          +-------------------+
          | Final seating and |
          | success checking  |
          +-------------------+
```

## 9. Illustration: Shelf Scenarios

```text
Figure 2. Canonical row scenarios

Legend:
| = upright book
/ = leaning book
[ ] = target insertion interval

1. Clear gap
   | | [   ] | |
   Direct insertion should usually succeed.

2. Tight gap
   | | [ ] | |
   Direct insertion may require contact adaptation.

3. Leaning neighbor
   | / [ ] | |
   Perception must detect reduced usable clearance.

4. Collapsed row
   | / [ ] / |
   Gap creation is likely required before insertion.
```

## 10. Hypotheses

This project tests the following hypotheses about the **full reshelving process**:

- **H1**: Successful reshelving across diverse shelf conditions requires a closed-loop mapping from perceived shelf state to corrective action, rather than a fixed insertion routine.
- **H2**: Policies that can both adjust the row and insert the book will achieve higher end-to-end success than insertion-only policies, especially in tight, leaning, and collapsed configurations.
- **H3**: A structured closed-loop system will be a practical way to realize this state-to-action mapping with the current setup, but the same task can later be reformulated as a more integrated policy.
- **H4**: VLA-style end-to-end models are promising for unifying perception and action, but they will likely need structured action representations, curriculum data, and strong baselines from the modular system before they can outperform a well-designed closed-loop controller in contact-rich shelf manipulation.

## 11. Experimental Scope

The primary task is robotic reshelving of a book into its correct position among neighboring shelf books. The robot identifies the slot using the spine label of the grasped book and the labels of books already on the shelf. RGB-D sensing is used to estimate the target slot and local row condition, and the robot may both adjust the row and insert the book as part of one closed-loop task.

Bookshelves are the main testbed, but the formulation is intentionally general enough to apply to other ordered-row insertion tasks such as filing folders in drawers or inserting items into tightly packed linear arrangements.

## 12. Experimental Design

### 12.1 Scenarios

Experiments will cover four difficulty levels:

- clear gap,
- tight gap,
- leaning neighbor,
- collapsed row.

Each scenario will be randomized over:

- lean angle,
- friction,
- book width and thickness,
- cover stiffness or compliance,
- neighbor compliance,
- small perception noise or pose estimation error.

### 12.2 Baselines

The evaluation should compare at least the following methods:

1. scripted insertion without row adjustment;
2. insertion-only residual policy;
3. closed-loop row adjustment plus scripted insertion;
4. closed-loop row adjustment plus residual insertion.

If feasible, a fifth comparison should be included:

5. an integrated policy that directly maps observations to actions across the full task, for example an end-to-end residual policy or a VLA-style policy.

These comparisons test the **entire loop**: how well each method maps observed state to action sequences that achieve the desired final shelf state.

### 12.3 Interpreting The Modular System

The structured system should not be treated as a separate problem from the full task. Instead, it is the first implementation of the full task under current constraints.

It is useful for three reasons:

1. it makes the overall reshelving loop testable now, with interpretable failure modes;
2. it reveals whether failures come from state estimation, action selection, or contact execution; and
3. it generates trajectories, successful corrections, and recovery behaviors that can later supervise or initialize a larger integrated model.

### 12.4 Path Toward VLA Integration

If the longer-term goal is a large integrated vision-language-action model, then the project can explicitly position the modular system as a stepping stone rather than a competing philosophy.

A sensible progression is:

1. build a closed-loop structured system that can solve the task reliably enough to collect meaningful demonstrations and rollout data;
2. represent the task as repeated transitions of the form:

```text
current shelf state -> chosen robot action -> new shelf state
```

3. use this data to train or fine-tune an integrated policy that conditions on RGB-D observations, language/task context, and recent action history;
4. compare whether the integrated model can recover the same adaptive behaviors as the structured controller, including deciding when to adjust the row before inserting.

In this framing, the VLA model is not only learning "insert the book." It is learning a policy over **state transitions toward a desired shelf configuration**.

## 13. Illustration: Comparison Matrix

```text
Figure 3. End-to-end policy comparison

Method                                      Closed-loop   Row adjust   Residual / learned
------------------------------------------------------------------------------------------
Scripted insertion only                     no            no           no
Insertion-only residual policy             limited       no           yes
Closed-loop adjust + scripted insertion    yes           yes          limited
Closed-loop adjust + residual insertion    yes           yes          yes
Integrated end-to-end policy / VLA         yes           implicit     yes
```

## 14. Evaluation Metrics

Success should require all of the following:

- correct placement in the intended slot,
- full seating of the target book,
- no neighbor displaced beyond a preset threshold,
- no neighbor toppled,
- no visible damage to the target book or its neighbors.

Additional metrics should include:

- task completion time,
- number of corrective actions,
- amount of row displacement introduced during gap creation,
- insertion force or contact effort if available,
- success as a function of perception error and scenario difficulty.

## 15. End-To-End Evaluation Protocol

To keep the work aligned with the full research question, each trial should be evaluated at the task level rather than the component level.

A complete trial consists of:

1. presenting the robot with an initial shelf state and a target book;
2. allowing the robot to observe, act, and re-observe for multiple closed-loop steps;
3. terminating when the goal is reached, the time budget is exceeded, or a safety or failure condition is triggered.

Primary reporting should include:

- end-to-end task success rate by scenario;
- average number of closed-loop action cycles per successful trial;
- fraction of cases in which row adjustment was invoked;
- success conditioned on initial row geometry and perception uncertainty;
- qualitative failure taxonomy, such as mislocalized slot, insufficient clearance creation, contact jam, neighbor toppling, or partial seating.

This protocol ensures that perception, action selection, and execution are judged by their contribution to the same end goal rather than as independent subsystems.

## 16. VLA Extension

The longer-term extension is to replace or absorb parts of the structured controller with a more integrated vision-language-action policy. In this proposal, a VLA model should be understood as a policy that predicts robot actions from multimodal context, not merely as a perception frontend.

### 16.1 What The VLA Would Learn

A VLA-style model for this task would learn mappings of the form:

```text
RGB-D shelf observation + target-book context + recent action history
    -> next robot action
```

The relevant action may be a discrete behavior token such as `adjust-row` or `insert`, a continuous motion command, or a structured combination of both.

### 16.2 Why The Structured System Comes First

Training a large integrated policy directly on this task is difficult because the environment is contact-rich, long-horizon, and safety-sensitive. The structured controller is therefore valuable as a data engine and baseline:

- it provides successful and failed closed-loop trajectories;
- it reveals useful action abstractions, such as when row adjustment must precede insertion;
- it supports curriculum design from easy clear-gap cases to harder collapsed rows;
- it offers a strong baseline against which any VLA policy must be compared.

### 16.3 Practical Integration Path

A realistic path toward VLA integration is:

1. implement the structured closed-loop system and collect trajectories across the benchmark scenarios;
2. annotate or derive compact action labels such as `insert`, `adjust-left`, `adjust-right`, `retry`, and `release`;
3. train an imitation or behavior-cloning model from observation histories to these structured actions;
4. fine-tune the policy with online data, residual learning, or human correction;
5. compare whether the integrated policy can recover the same adaptive behavior and end-to-end success.

This makes the VLA direction a continuation of the same research question rather than a separate project.

## 17. Expected Contributions

The expected contributions are:

1. a closed-loop formulation of robotic reshelving as state estimation, action selection, and state transformation toward a desired final shelf configuration;
2. a practical structured controller that makes the full task testable under current hardware and data limitations;
3. an empirical analysis of which initial shelf states require row adjustment and how much residual learning improves end-to-end performance;
4. a concrete path for scaling from an interpretable structured controller to a more integrated VLA-style policy.

## 18. Expected Outcome

The most likely outcome is that generalizable robotic reshelving will require treating the problem as a closed-loop state-to-action process rather than as isolated perception and insertion components. In the near term, a structured controller will likely be the most practical way to realize this loop and study its failure modes. In the longer term, the same formulation should support a more integrated VLA-style model that learns to predict robot actions from multimodal observations while still targeting the same desired final shelf state.

## 19. Concise Proposal Summary

This project investigates robotic book reshelving as a closed-loop problem in which the robot observes the current shelf state, chooses actions that modify that state, and aims to reach a desired final state where the target book is correctly inserted and the row remains stable. The immediate research approach uses a structured controller to make this full process testable and interpretable, while the longer-term goal is to explore whether the same state-to-action mapping can be learned in a more integrated VLA-style model.

# Pole Hunting With VLMs

This repo is an interview task for building a small vision-language agent that can explore street-level panoramas and find poles of interest.

The goal is not to produce a giant production system. The goal is to see how you think, how you use tools, how you make tradeoffs, and how we work together on a practical AI problem.

## Task

Build an agent that navigates the streets of Bhelupur, locating and classifying poles of interest.

Therer are 5 poles on the map, and the agent should be able to, in one run, locate and classify all of them.

Target pole types:

- Distribution transformer: Two pole structure, with a large transformer between the two poles.
- Lamp post: Single pole structure, with a lamp on the top.
- Billboard pole: Single pole structure, with a billboard on the top.
- Low Tension pole: low tension: Single pole structure, with a low tension wire running through the pole.

At any point in time, the agent should be bound to a single panorama, and is not allowed to move between panoramas that are further away than 20 meters. 
With that being said, the connectivity of the panoramas is not fixed; in fact, rewiring them to improve ease of mobility is recommended.

## Recommended Design

We strongly recommend a discrete model of states and actions.

For discrete turning, we recommend rendering panorama tiles or fixed camera crops so the model sees a controlled set of directions instead of an unconstrained 360-degree view or dynamic slices. For movement, the panorama sequence can be treated as a route: "forward" and "backward" move along that sequence, with the current orientation defining what forward means visually. There are also other, more sophisticated ways to model the navigation state, but this is a good starting point.

A simple state might include the current panorama id, current orientation bin, recent visual evidence, and any current classification hypothesis. A simple action set might be `turn_left`, `turn_right`, `move_forward`, `move_backward`, and `classify_or_stop`.

The agent should choose an action from the current state. Qwen3-VL-4B-Instruct should be sufficient for this task and should fit on a 40 GB A100, though you may use any reasonable model stack you prefer.

## Deliverables

- A demo, ideally a live run of the agent solving the task.
- Environment: state/action space design.
- Agent: Agent loop, skills and prompt (context) design.
- Code and any artifacts needed to run or inspect the solution.
- A short explanation of the major tradeoffs you made.

Be ready to discuss:

- How would you make the system more accurate with more time?
- How would you make it faster or cheaper with more time?
- How would your design change if you also had to detect poles, not only navigate to known or candidate poles?

## Starter App

Included in this repo:

- React/Vite UI with a Leaflet map and Pannellum panorama walkthrough.
- Static Node server for local metadata and panorama images.
- Expected local data layout for the Bhelupur tiny run:
  - `data/metadata/panoramas.json`: panorama positions, route/session ordering, headings, dimensions, and image paths.
  - `data/metadata/poles.geojson`: curated inferred pole coordinates.
  - `data/metadata/bhelupur-tiny.geojson`: area boundary.
  - `data/panoramas/`: panorama images.

The `data/` directory is intentionally not checked in.

## Data Setup

Unzip the panorama bundle so files land under:

```text
data/panoramas/SN_UNKNOWN/...
```

The app expects each panorama path to match the `imagePath` field in `data/metadata/panoramas.json`.

Check that local imagery is present:

```bash
npm run check-data
```

## Run

Install dependencies:

```bash
npm install
```

Start the backend and frontend together:

```bash
make dev
```

Default URLs:

- Frontend: `http://127.0.0.1:5177`
- Backend: `http://127.0.0.1:8787`

You can also run each side separately:

```bash
make backend
make frontend
```

## Notes

AI tooling is table stakes. Use it as much as you want.

That said, human judgment matters more than ever here. Please be prepared to explain your design, what is happening under the hood, where the model is likely to fail, and why your approach is a good fit for this task.

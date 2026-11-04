# BiasLens (analysis UI)

Next.js frontend for the bias-analysis tool (Input -> Detect -> Mitigate -> Evaluate), wired to the
14-category LoRA-fine-tuned Qwen2.5 severity model in `../src/training/` via a local FastAPI service.

Start the detection service before `npm run dev`, from the repo root:

```bash
python -m uvicorn serve:app --reload --port 8000 --app-dir src/training
```

Use `python -m uvicorn` rather than a bare `uvicorn` if this machine has more than one Python
installation (e.g. Anaconda) -- a bare `uvicorn` on PATH can resolve to one that's missing
`torch`/`transformers`/`peft`.

See the root `README.md` for full setup (dataset, training, evaluation). Without the FastAPI service
running, the Detect/Evaluate stages will show a "service unreachable" error.

## Project structure

- `app/` -- Next.js pages: `page.tsx` (the analysis tool) and `api/*` (proxy routes to FastAPI)
- `components/analysis/` -- the four pipeline stages (Input, Detect, Mitigate, Evaluate)
- `components/ui/` -- shared UI primitives (Button, RadarChart)
- `lib/` -- API client and category-label constants
- `styles/` -- global CSS and Tailwind configuration
- `types/` -- TypeScript types for the detection/mitigation/evaluation API shapes

## Setup

```bash
npm install
npm run dev
```

Open http://localhost:3000.

## Useful commands

- `npm run dev` -- start Next.js in development mode
- `npm run build` -- build for production
- `npm run start` -- start the production build
- `npm run lint` -- run lint checks
- `npm run type-check` -- run TypeScript type checking

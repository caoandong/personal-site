This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.

## Return of the King

The website serves the game at `/return-of-the-king`. The website commit pins a SHA-256-verified, self-contained HTTP distribution. All player requests stay on the website origin.

To update the game, export its optimized HTTP distribution from the game source tree. Run `python3 scripts/import-game.py /path/to/dist` with Python 3 and FFmpeg installed. The importer creates AAC delivery copies, checks their durations, prepares the game HTML for its subpath and writes an immutable release directory and pin. Keep the preceding release directory available for players who already opened the game.

Open a website PR with the new directory and `game/release.json`. CI verifies the digest, lints, builds and plays through the opening in Chrome, then exercises a later scene's deferred score. Vercel deploys the merged website. The production deployment event runs the same browser check on the public game. To roll back only the game, restore the preceding release pin in a PR while retaining both directories.

### Game startup and verification

The initial page presents the in-game portrait and binds Play and sound before the 3D engine downloads. In a new browser, Enable sound starts the streamed title score. Play immediately opens a short courtyard approach while the movement pack downloads. Reduced-motion and data-saver sessions use its first frame. Download progress comes from the movement response body. The game starts as soon as its opening assets are ready and refines the baked lighting after the first render.

The web packer lives in the game's `experiments/nvidia-sifu-level/scripts/web-release/tools/build.ts`. Build the current portable inputs first, then run the web packer with Bun, passing the game directory and a Pillow-enabled Python interpreter. Import its `web-release/dist` output with the website importer. The shipped bundle includes its own workers, UI sounds, preview lighting, title portrait and courtyard film.

The browser check deliberately holds the movement download until it has observed a non-silent title soundtrack and an advancing opening movie. It then releases the response and verifies the handoff to CS-01, combat, full-resolution lighting, and deferred music.

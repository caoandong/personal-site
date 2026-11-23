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

The entry page paints a dark text loader while the lead character and title score load. The title appears after the first live 3D portrait frame. Music streams automatically where browser policy permits, and Enable sound unlocks it on a fresh visit. Start game stays disabled while the opening scene loads. Once it is ready, the player chooses when to enter. Stage changes are announced to screen readers and the loading indicator respects reduced motion.

The web packer lives in the game's `experiments/nvidia-sifu-level/scripts/web-release/tools/build.ts`. Build the current portable inputs first, then run the web packer with Bun, passing the game directory and a Pillow-enabled Python interpreter. Import its `web-release/dist` output with the website importer. The packer extracts the lead's title idle from the game animation data so the animated portrait can be drawn before the campaign movement pack arrives.

The Chrome check holds the engine bundle to sample the dark first frame. It then holds the movement download while it checks for skinned geometry on the game canvas, audible title music and a disabled Start button. After the download resumes, it verifies that a fresh press of Start enters CS-01. It also checks combat, full-resolution lighting and deferred music.

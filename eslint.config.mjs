import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

// NOTE: eslint-plugin-tailwindcss is installed but disabled until it supports Tailwind v4.
// The plugin uses resolveConfig which was removed in Tailwind v4's CSS-first approach.
// See: https://github.com/francoismassart/eslint-plugin-tailwindcss/issues
// To enable when supported, uncomment the tailwindcss import and config below.

// import tailwindcss from "eslint-plugin-tailwindcss";

const eslintConfig = defineConfig([
  ...nextVitals,
  ...nextTs,
  // Tailwind CSS rules (disabled until v4 support)
  // ...tailwindcss.configs["flat/recommended"],
  // {
  //   settings: {
  //     tailwindcss: {
  //       callees: ["cn", "cva", "clsx"],
  //       whitelist: ["prose-.*", "not-prose", "group-hover:.*"],
  //     },
  //   },
  //   rules: {
  //     "tailwindcss/no-custom-classname": "warn",
  //     "tailwindcss/classnames-order": "warn",
  //     "tailwindcss/no-contradicting-classname": "error",
  //   },
  // },
  // Override default ignores of eslint-config-next.
  globalIgnores([
    // Default ignores of eslint-config-next:
    ".next/**",
    "out/**",
    "build/**",
    "next-env.d.ts",
  ]),
]);

export default eslintConfig;

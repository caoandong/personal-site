import { readFile } from 'node:fs/promises'
import rehypePrettyCode from 'rehype-pretty-code'
import { createHighlighter } from 'shiki'

export const nvidiaSass = {
  ...JSON.parse(
    await readFile(
      new URL(
        '../../extensions/nvidia-sass/syntaxes/nvidia-sass.tmLanguage.json',
        import.meta.url
      ),
      'utf8'
    )
  ),
  name: 'nvidia-sass',
  displayName: 'NVIDIA SASS',
  aliases: ['cuda-sass', 'sassasm'],
}

export default function rehypeCode(options) {
  return rehypePrettyCode({
    ...options,
    getHighlighter: (highlighterOptions) =>
      createHighlighter({
        ...highlighterOptions,
        langs: [...highlighterOptions.langs, nvidiaSass],
      }),
  })
}

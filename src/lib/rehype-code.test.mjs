import assert from 'node:assert/strict'
import test from 'node:test'
import { createHighlighter } from 'shiki'
import { nvidiaSass } from './rehype-code.mjs'

test('NVIDIA SASS tokens keep their semantic scopes', async (t) => {
  const highlighter = await createHighlighter({
    themes: ['github-light'],
    langs: [nvidiaSass],
  })
  t.after(() => highlighter.dispose())

  const { tokens } = highlighter.codeToTokens(
    'LDG.E R2, desc[UR4][R2.64] ; // A[tid]',
    {
      lang: 'nvidia-sass',
      theme: 'github-light',
      includeExplanation: 'scopeName',
    }
  )
  const scopes = tokens
    .flatMap((line) => line.flatMap((token) => token.explanation ?? []))
    .flatMap(({ content, scopes }) =>
      scopes.map(({ scopeName }) => [content, scopeName])
    )

  for (const expected of [
    ['LDG', 'keyword.control.instruction.nvidia-sass'],
    ['R2', 'variable.other.register.nvidia-sass'],
    ['desc', 'storage.type.memory-space.nvidia-sass'],
    ['// A[tid]', 'comment.line.double-slash.nvidia-sass'],
  ]) {
    assert.ok(
      scopes.some(
        ([content, scope]) => content === expected[0] && scope === expected[1]
      )
    )
  }
})

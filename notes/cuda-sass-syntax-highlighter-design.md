# NVIDIA CUDA SASS syntax highlighter: research and design

Research date: 2026-07-18

## Decision

Build a dependency-free VS Code language extension around one JSON TextMate grammar. Do not start with a parser, language server, semantic-token provider, custom theme, or opcode generator.

SASS disassembly is line-oriented and structurally regular, which is exactly where TextMate works well. VS Code uses TextMate as its primary syntax-tokenization engine and layers semantic tokens on top only when project-level understanding is useful. The same grammar can also be loaded by Shiki, so one source can highlight both VS Code files and this site's MDX code blocks.

The extension should support official text emitted by `nvdisasm` and `cuobjdump -sass`, including full ELF dumps, code-only output, source-line annotations, instruction encodings, register life-range comments, and current Blackwell operands.

## Why this approach

| Approach                         | Verdict    | Reason                                                                                                                                         |
| -------------------------------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| TextMate grammar                 | Build this | Native VS Code path, instant incremental coloring, no runtime code, and portable to Shiki.                                                     |
| Semantic token provider          | Defer      | It adds extension code and theme-dependent behavior but cannot improve the basic lexical roles enough to justify it.                           |
| Tree-sitter or custom parser     | Skip       | NVIDIA documents the disassembly format and opcode sets, not a stable assembler grammar; highlighting does not require an AST.                 |
| Language server                  | Skip       | There are no declarations, types, imports, or project-wide symbol resolution that a syntax-only extension needs.                               |
| Reuse a generic assembly grammar | Skip       | SASS predicates, uniform registers, constant banks, descriptor operands, tensor memory, relocations, and ELF directives need their own scopes. |

This follows VS Code's own architecture: TextMate grammars provide syntax tokens, while semantic providers are an optional addition for information that requires deeper language knowledge ([VS Code syntax highlighting guide](https://code.visualstudio.com/api/language-extensions/syntax-highlight-guide), [semantic highlighting guide](https://code.visualstudio.com/api/language-extensions/semantic-highlight-guide)). TextMate evaluates regular expressions one line at a time, which fits SASS instruction records and keeps incremental retokenization cheap ([TextMate grammar manual](https://macromates.com/manual/en/language_grammars)).

## Input dialect

The supported language is **NVIDIA disassembly text**, not a third-party writable SASS assembler dialect.

Cover these official forms:

- `nvdisasm` full output: ELF directives, sections, labels, data records, and instructions.
- `nvdisasm -c`: code sections only.
- `nvdisasm -g`, `-gi`, and `-gp`: CUDA or PTX source annotations.
- `nvdisasm -hex`: trailing instruction-encoding comments.
- `nvdisasm -plr`: trailing register life-range tables.
- `cuobjdump -sass`: fatbin headers, functions, instructions, and two-line encodings.
- Turing through Blackwell operands, including `R`, `UR`, `SR`, `P`, `UP`, `c[][]`, `desc[][]`, `gdesc[]`, `idesc[]`, and `tmem[]`.

NVIDIA's current Binary Utilities documentation defines the instruction/operand shape, lists architecture-specific opcode sets, and shows the output variants above. It also exposes `nvdisasm --emit-json`, whose instruction objects separate `predicate`, `opcode`, and `operands`; that is useful as a corpus oracle, but should not be a runtime dependency ([CUDA Binary Utilities 13.3](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html)).

Non-goals for version 1:

- PTX, CUDA C++, AMD assembly, or indented Sass stylesheets.
- Validating whether an opcode or modifier exists on a selected SM architecture.
- Hover documentation, control-flow graphs, occupancy analysis, or disassembling binaries.
- Hand-authored CuAssembler/maxas scheduling syntax.

## Naming and file association

Do not use the generic language ID `sass` or root scope `source.sass`. Existing extensions already use both for the indented Sass stylesheet language, and a current GPU disassembly extension also claims them for NVIDIA SASS. That collision makes theme rules, fenced-code aliases, and `.sass` detection unpredictable.

Use:

- Language ID: `nvidia-sass`
- Display aliases: `NVIDIA SASS`, `CUDA SASS`
- TextMate root scope: `source.nvidia-sass`
- Markdown/Shiki aliases: `nvidia-sass`, `cuda-sass`, `sassasm`
- Unambiguous suffixes: `.nv.sass`, `.cubin.sass`, `.sassasm`, `.sassdump`

Do **not** claim every `.sass` file by default. Users who keep `nvdisasm` output as `kernel.sass` can select **NVIDIA SASS** once through Change Language Mode or add a workspace association:

```json
{
  "files.associations": {
    "*.sass": "nvidia-sass"
  }
}
```

VS Code supports language IDs, aliases, suffixes, filename patterns, and first-line detection through `contributes.languages`; user associations are the native override mechanism ([language contribution point](https://code.visualstudio.com/api/references/contribution-points#contributes.languages), [language overview](https://code.visualstudio.com/docs/languages/overview)).

## Minimal extension shape

```text
nvidia-sass/
├── package.json
├── README.md
├── LICENSE
├── syntaxes/
│   └── nvidia-sass.tmLanguage.json
└── test/
    └── syntax.test.nv.sass
```

There is no `main`, activation event, TypeScript build, or production dependency.

The manifest should be approximately:

```json
{
  "name": "nvidia-sass",
  "displayName": "NVIDIA SASS",
  "version": "0.1.0",
  "engines": { "vscode": "^1.85.0" },
  "categories": ["Programming Languages"],
  "contributes": {
    "languages": [
      {
        "id": "nvidia-sass",
        "aliases": ["NVIDIA SASS", "CUDA SASS"],
        "extensions": [".sassasm", ".sassdump"],
        "filenamePatterns": ["*.nv.sass", "*.cubin.sass"],
        "firstLine": "^\\s*(?:Fatbin elf code:|code for sm_[0-9]+[a-z]?|\\.headerflags\\s+@\"EF_CUDA_)"
      }
    ],
    "grammars": [
      {
        "language": "nvidia-sass",
        "scopeName": "source.nvidia-sass",
        "path": "./syntaxes/nvidia-sass.tmLanguage.json"
      }
    ]
  }
}
```

VS Code loads JSON TextMate grammars directly through `contributes.grammars`; a generated YAML source is unnecessary while the grammar remains small ([grammar contribution point](https://code.visualstudio.com/api/references/contribution-points#contributes.grammars)).

## Token model

Use standard TextMate scope families so every normal VS Code theme works. Do not ship colors with the language extension.

| SASS element               | Example                                | Scope                                               |
| -------------------------- | -------------------------------------- | --------------------------------------------------- |
| Instruction record         | whole line                             | `meta.instruction.nvidia-sass`                      |
| Instruction address        | `/*01c0*/`                             | `constant.numeric.address.nvidia-sass`              |
| Predicate guard            | `@!P0`, `@UP3`                         | `keyword.control.predicate.nvidia-sass`             |
| Opcode                     | `FADD`, `UTCHMMA`                      | `keyword.control.instruction.nvidia-sass`           |
| Opcode/operand modifier    | `.FTZ`, `.SYS`, `.reuse`, `.x32`       | `keyword.other.modifier.nvidia-sass`                |
| General/uniform register   | `R17`, `RZ`, `UR4`                     | `variable.other.register.nvidia-sass`               |
| Predicate register         | `P0`, `PT`, `UP1`, `UPT`               | `variable.other.register.predicate.nvidia-sass`     |
| Special register           | `SR_TID.X`                             | `support.constant.system-register.nvidia-sass`      |
| Memory-space introducer    | `c`, `desc`, `gdesc`, `idesc`, `tmem`  | `storage.type.memory-space.nvidia-sass`             |
| Label definition           | `.L_1:`, `vector_add:`                 | `entity.name.label.nvidia-sass`                     |
| Label/call target          | `` `(.L_1) ``                          | `variable.other.label.nvidia-sass`                  |
| ELF directive              | `.section`, `.align`, `.word`          | `keyword.other.directive.nvidia-sass`               |
| Relocation operator        | `@lo`, `@hi`, `@srel`, `index@`        | `keyword.operator.relocation.nvidia-sass`           |
| Immediate                  | `0x118`, `-0.5`, `2.3e-07`             | `constant.numeric.nvidia-sass`                      |
| Metadata string            | `@"EF_CUDA_SM100"`                     | `string.quoted.double.nvidia-sass`                  |
| Source/liveness annotation | `//## File ...`, `// liveness columns` | `comment.line.double-slash.nvidia-sass`             |
| Encoding                   | `/* 0x000e... */`                      | `constant.numeric.instruction-encoding.nvidia-sass` |

The visual hierarchy should be role-based: opcode strongest; guard and labels next; registers and memory spaces distinct; immediates normal; addresses, encodings, punctuation, and generated metadata quieter. Themes decide the actual colors.

## Grammar strategy

Order the top-level rules from most contextual to most generic:

1. Source annotations and `//` comments.
2. Instruction lines.
3. Directive lines.
4. Label definitions.
5. Metadata strings.
6. Shared operand atoms: memory spaces, registers, relocations, numbers, operators, and punctuation.

Match opcodes **by position and shape**, not from an enumerated list. A representative instruction prefix is:

```regex
^(\s*)(?:(/\*)([0-9A-Fa-f]+)(\*/)\s*)?(@!?U?P(?:T|[0-9]+)\s+)?([A-Z][A-Z0-9_]*)(?=\.|\s|;)
```

That recognizes an optional address, optional scalar or uniform predicate, and the opcode only where an instruction can begin. It automatically handles future opcodes without recoloring arbitrary uppercase ELF metadata as instructions.

Inside an instruction record, reuse small repository rules for:

```regex
# registers
\b(?:UR(?:Z|[0-9]+)|R(?:Z|[0-9]+))\b

# predicate registers
\b(?:UP(?:T|[0-9]+)|P(?:T|[0-9]+))\b

# special registers, including components
\bSR_[A-Z0-9_]+(?:\.[A-Za-z0-9_]+)*\b

# current memory/descriptor forms
\b(?:c|desc|gdesc|idesc|tmem)(?=\[)

# signed hexadecimal, integer, decimal, and scientific immediates
(?<![A-Za-z0-9_])-?(?:0[xX][0-9A-Fa-f]+|(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)(?![A-Za-z0-9_])
```

Do not wrap address and encoding fields in a broad `comment.block` scope. Although NVIDIA prints them with `/* */`, they are machine-readable metadata; scope only their delimiters as comment punctuation and keep the value numeric. Ordinary free-form block comments can still use `comment.block`.

Avoid nested unbounded quantifiers, catch-all `.*` before meaningful alternatives, and one giant instruction regex. TextMate runs in the editor renderer, so short anchored rules are safer and easier to test.

## What existing tooling gets wrong

The current `gpu-disasm` marketplace extension proves that a grammar-only SASS mode is viable, but its 44-line grammar is only a baseline. It:

- uses the conflicting `sass` language ID and `source.sass` scope;
- colors any uppercase word as an opcode, regardless of position;
- treats instruction addresses and encodings as undifferentiated comments;
- misses uniform predicate guards, labels, directives, relocations, descriptor/tensor-memory forms, and special-register components.

Those are design lessons, not reasons to add a parser. Contextual line anchors and a few shared lexical rules cover the gaps ([current grammar source](https://github.com/smashedpumpkin/gpu-disasm/blob/main/syntaxes/sass.tmLanguage.json), [marketplace entry](https://marketplace.visualstudio.com/items?itemName=MattGordon.gpu-dism)).

## Test plan

Keep one scope-assertion corpus, covering:

- the repository's existing SM80 `vector_add.sass` constructs;
- predicated branches and absolute-value operands;
- `cuobjdump` two-line instruction encodings;
- `nvdisasm` source annotations and life-range comments;
- relocations such as `32@lo(...)` and `@srel`;
- Hopper `desc`/`gdesc` and Blackwell `idesc`/`tmem` operands;
- lowercase modifiers such as `.reuse` and `.x32`;
- negative tests proving uppercase metadata and Sass stylesheet text are not opcodes.

Use `vscode-tmgrammar-test` for inline scope assertions; Microsoft's `vscode-textmate` package explicitly recommends it to grammar authors ([vscode-textmate](https://www.npmjs.com/package/vscode-textmate)). Then inspect the corpus manually with **Developer: Inspect Editor Tokens and Scopes** in Dark+, Light+, and a high-contrast theme.

Acceptance criteria:

1. Every construct above has the intended scope and no rule leaks into the next line.
2. Unknown future-looking opcodes and modifiers still receive structural instruction scopes.
3. A large real dump tokenizes with approximately linear scaling and no Oniguruma timeout.
4. The grammar renders correctly in VS Code and Shiki from the same JSON artifact.
5. Plain `.sass` stylesheet files are not claimed automatically.

## Site integration

This repository already uses Shiki through `rehype-pretty-code`. Shiki accepts a custom TextMate grammar object and language aliases, so the extension grammar can be registered as `nvidia-sass`/`cuda-sass` and used in MDX fences without a second tokenizer ([Shiki custom-language guide](https://shiki.style/guide/load-lang)).

Use:

````markdown
```nvidia-sass
/*00a0*/ @!P0 LDG.E.SYS R2, desc[UR4][R2.64] ;
/*00b0*/      LDTM.x32 R20, tmem[UR7] ;
```
````

Keep `asm` available for generic teaching pseudocode. Use `nvidia-sass` only for actual NVIDIA disassembly.

## Build order

1. Create the five-file grammar-only extension.
2. Implement the instruction-line rule and shared operand rules against one mixed-architecture test corpus.
3. Verify scopes in VS Code's inspector and run the scope test.
4. Register the same JSON grammar with Shiki in this site.
5. Publish only after `.sass` collision behavior and light/dark/high-contrast rendering are checked.

Add a semantic provider only if a later feature needs context that regex cannot know, such as architecture-invalid instructions, read/write operand roles, branch-target navigation, or hover documentation. If that day comes, consume `nvdisasm --emit-json` or a small line parser; do not replace the TextMate fallback.

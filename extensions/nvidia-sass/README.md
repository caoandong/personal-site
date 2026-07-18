# NVIDIA SASS

Syntax highlighting for NVIDIA CUDA SASS disassembly emitted by `nvdisasm` and `cuobjdump -sass`.

The extension highlights instructions, predicates, modifiers, registers, constant banks, Hopper/Blackwell descriptors and tensor memory, labels, relocations, ELF directives, source annotations, addresses, and instruction encodings.

It registers `.sass`, `.nv.sass`, `.cubin.sass`, `.sassasm`, and `.sassdump` files as **NVIDIA SASS**. This intentionally treats `.sass` as NVIDIA assembly rather than the indented Sass stylesheet language.

The extension contains no activation code or runtime dependencies.

## Test

```sh
npm install
npm test
```

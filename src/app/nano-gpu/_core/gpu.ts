export type MachineIr = {
  smCount: number
  warpSize: number
  registersPerLane: number
  sharedMemoryWords: number
  hbmWords: number
  l2Words: number
}

export type Instruction =
  | { op: 'move_immediate'; destination: number; value: number }
  | { op: 'lane_id'; destination: number }
  | { op: 'warp_id'; destination: number }
  | { op: 'add'; destination: number; left: number; right: number }
  | {
      op: 'multiply_immediate'
      destination: number
      source: number
      value: number
    }
  | { op: 'load_global'; destination: number; address: number }
  | { op: 'store_global'; address: number; source: number }
  | { op: 'load_shared'; destination: number; address: number }
  | { op: 'store_shared'; address: number; source: number }
  | { op: 'exit' }

export type LaneEffect = {
  laneId: number
  registerWrite?: { register: number; value: number }
  memoryRead?: {
    space: 'global' | 'shared'
    address: number
    value: number
    source: 'hbm' | 'l2' | 'shared'
  }
  memoryWrite?: {
    space: 'global' | 'shared'
    address: number
    value: number
  }
  exit?: true
}

export type TraceEvent = {
  cycle: number
  smId: number
  warpId: number
  pc: number
  instruction: Instruction
  activeLanes: number[]
  effects: LaneEffect[]
}

function checkSize(name: string, value: number) {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${name} must be a positive integer`)
  }
}

function checkIndex(name: string, index: number, size: number) {
  if (!Number.isInteger(index) || index < 0 || index >= size) {
    throw new Error(`${name} ${index} is outside 0..${size - 1}`)
  }
}

export class Hbm {
  private readonly words: Uint32Array

  constructor(wordCount: number) {
    checkSize('HBM size', wordCount)
    this.words = new Uint32Array(wordCount)
  }

  read(address: number) {
    checkIndex('HBM address', address, this.words.length)
    return this.words[address]
  }

  write(address: number, value: number) {
    checkIndex('HBM address', address, this.words.length)
    this.words[address] = value
  }

  snapshot() {
    return Array.from(this.words)
  }
}

export class L2Cache {
  readonly hbm: Hbm
  readonly capacity: number
  hits = 0
  misses = 0
  private readonly lines = new Map<number, number>()

  constructor(hbm: Hbm, capacity: number) {
    checkSize('L2 size', capacity)
    this.hbm = hbm
    this.capacity = capacity
  }

  read(address: number) {
    if (this.lines.has(address)) {
      const value = this.lines.get(address) as number
      this.hits += 1
      this.remember(address, value)
      return { value, source: 'l2' as const }
    }

    const value = this.hbm.read(address)
    this.misses += 1
    this.remember(address, value)
    return { value, source: 'hbm' as const }
  }

  write(address: number, value: number) {
    this.hbm.write(address, value)
    this.remember(address, value)
  }

  snapshot() {
    return {
      hits: this.hits,
      misses: this.misses,
      lines: Array.from(this.lines.entries()).map(([address, value]) => ({
        address,
        value,
      })),
    }
  }

  private remember(address: number, value: number) {
    this.lines.delete(address)
    this.lines.set(address, value)

    if (this.lines.size > this.capacity) {
      const oldestAddress = this.lines.keys().next().value
      if (oldestAddress !== undefined) this.lines.delete(oldestAddress)
    }
  }
}

export class SharedMemory {
  private readonly words: Uint32Array

  constructor(wordCount: number) {
    checkSize('shared memory size', wordCount)
    this.words = new Uint32Array(wordCount)
  }

  read(address: number) {
    checkIndex('shared memory address', address, this.words.length)
    return this.words[address]
  }

  write(address: number, value: number) {
    checkIndex('shared memory address', address, this.words.length)
    this.words[address] = value
  }

  snapshot() {
    return Array.from(this.words)
  }
}

export class RegisterFile {
  private readonly values: Uint32Array

  constructor(
    readonly laneCount: number,
    readonly registersPerLane: number
  ) {
    checkSize('lane count', laneCount)
    checkSize('register count', registersPerLane)
    this.values = new Uint32Array(laneCount * registersPerLane)
  }

  read(lane: number, register: number) {
    return this.values[this.offset(lane, register)]
  }

  write(lane: number, register: number, value: number) {
    this.values[this.offset(lane, register)] = value
  }

  snapshot() {
    return Array.from({ length: this.laneCount }, (_, lane) =>
      Array.from({ length: this.registersPerLane }, (_, register) =>
        this.read(lane, register)
      )
    )
  }

  private offset(lane: number, register: number) {
    checkIndex('lane', lane, this.laneCount)
    checkIndex('register', register, this.registersPerLane)
    return lane * this.registersPerLane + register
  }
}

export class Warp {
  pc = 0
  readonly activeMask: boolean[]
  readonly registers: RegisterFile

  constructor(
    readonly id: number,
    readonly program: readonly Instruction[],
    laneCount: number,
    registersPerLane: number
  ) {
    this.activeMask = Array.from({ length: laneCount }, () => true)
    this.registers = new RegisterFile(laneCount, registersPerLane)
  }

  get done() {
    return this.pc >= this.program.length || !this.activeMask.some(Boolean)
  }

  snapshot() {
    return {
      id: this.id,
      pc: this.pc,
      done: this.done,
      activeMask: [...this.activeMask],
      registers: this.registers.snapshot(),
    }
  }
}

export class CudaCore {
  execute(instruction: Instruction, warp: Warp, laneId: number): LaneEffect {
    const read = (register: number) => warp.registers.read(laneId, register)

    switch (instruction.op) {
      case 'move_immediate':
        return {
          laneId,
          registerWrite: {
            register: instruction.destination,
            value: instruction.value,
          },
        }
      case 'lane_id':
        return {
          laneId,
          registerWrite: {
            register: instruction.destination,
            value: laneId,
          },
        }
      case 'warp_id':
        return {
          laneId,
          registerWrite: {
            register: instruction.destination,
            value: warp.id,
          },
        }
      case 'add':
        return {
          laneId,
          registerWrite: {
            register: instruction.destination,
            value: read(instruction.left) + read(instruction.right),
          },
        }
      case 'multiply_immediate':
        return {
          laneId,
          registerWrite: {
            register: instruction.destination,
            value: read(instruction.source) * instruction.value,
          },
        }
      case 'exit':
        return { laneId, exit: true }
      default:
        throw new Error(`${instruction.op} must run on the load/store unit`)
    }
  }
}

export class LoadStoreUnit {
  execute(
    instruction: Instruction,
    warp: Warp,
    laneId: number,
    l2: L2Cache,
    sharedMemory: SharedMemory
  ): LaneEffect {
    const read = (register: number) => warp.registers.read(laneId, register)

    switch (instruction.op) {
      case 'load_global': {
        const address = read(instruction.address)
        const result = l2.read(address)
        return {
          laneId,
          registerWrite: {
            register: instruction.destination,
            value: result.value,
          },
          memoryRead: {
            space: 'global',
            address,
            value: result.value,
            source: result.source,
          },
        }
      }
      case 'store_global':
        return {
          laneId,
          memoryWrite: {
            space: 'global',
            address: read(instruction.address),
            value: read(instruction.source),
          },
        }
      case 'load_shared': {
        const address = read(instruction.address)
        const value = sharedMemory.read(address)
        return {
          laneId,
          registerWrite: { register: instruction.destination, value },
          memoryRead: {
            space: 'shared',
            address,
            value,
            source: 'shared',
          },
        }
      }
      case 'store_shared':
        return {
          laneId,
          memoryWrite: {
            space: 'shared',
            address: read(instruction.address),
            value: read(instruction.source),
          },
        }
      default:
        throw new Error(`${instruction.op} is not a memory instruction`)
    }
  }
}

export class WarpScheduler {
  private cursor = -1

  select(warps: readonly Warp[]) {
    for (let offset = 1; offset <= warps.length; offset += 1) {
      const index = (this.cursor + offset) % warps.length
      if (!warps[index].done) {
        this.cursor = index
        return warps[index]
      }
    }

    return undefined
  }
}

function isMemoryInstruction(instruction: Instruction) {
  return (
    instruction.op === 'load_global' ||
    instruction.op === 'store_global' ||
    instruction.op === 'load_shared' ||
    instruction.op === 'store_shared'
  )
}

export class StreamingMultiprocessor {
  readonly sharedMemory: SharedMemory
  readonly scheduler = new WarpScheduler()
  readonly cudaCores: CudaCore[]
  readonly loadStoreUnit = new LoadStoreUnit()
  readonly warps: Warp[] = []

  constructor(
    readonly id: number,
    readonly warpSize: number,
    sharedMemoryWords: number
  ) {
    this.sharedMemory = new SharedMemory(sharedMemoryWords)
    this.cudaCores = Array.from({ length: warpSize }, () => new CudaCore())
  }

  get done() {
    return this.warps.every((warp) => warp.done)
  }

  addWarp(warp: Warp) {
    this.warps.push(warp)
  }

  tick(cycle: number, l2: L2Cache): TraceEvent | undefined {
    const warp = this.scheduler.select(this.warps)
    if (!warp) return undefined

    const pc = warp.pc
    const instruction = warp.program[pc]
    const activeLanes = warp.activeMask.flatMap((active, lane) =>
      active ? [lane] : []
    )
    const effects = activeLanes.map((laneId) =>
      isMemoryInstruction(instruction)
        ? this.loadStoreUnit.execute(
            instruction,
            warp,
            laneId,
            l2,
            this.sharedMemory
          )
        : this.cudaCores[laneId].execute(instruction, warp, laneId)
    )

    for (const effect of effects) {
      if (effect.registerWrite) {
        warp.registers.write(
          effect.laneId,
          effect.registerWrite.register,
          effect.registerWrite.value
        )
      }

      if (effect.memoryWrite?.space === 'global') {
        l2.write(effect.memoryWrite.address, effect.memoryWrite.value)
      } else if (effect.memoryWrite?.space === 'shared') {
        this.sharedMemory.write(
          effect.memoryWrite.address,
          effect.memoryWrite.value
        )
      }

      if (effect.exit) warp.activeMask[effect.laneId] = false
    }

    warp.pc += 1
    return {
      cycle,
      smId: this.id,
      warpId: warp.id,
      pc,
      instruction,
      activeLanes,
      effects,
    }
  }

  snapshot() {
    return {
      id: this.id,
      sharedMemory: this.sharedMemory.snapshot(),
      warps: this.warps.map((warp) => warp.snapshot()),
    }
  }
}

export class Gpu {
  readonly hbm: Hbm
  readonly l2: L2Cache
  readonly sms: StreamingMultiprocessor[]
  cycle = 0
  private launched = false

  constructor(readonly machine: MachineIr) {
    for (const [name, value] of Object.entries(machine)) checkSize(name, value)

    this.hbm = new Hbm(machine.hbmWords)
    this.l2 = new L2Cache(this.hbm, machine.l2Words)
    this.sms = Array.from(
      { length: machine.smCount },
      (_, id) =>
        new StreamingMultiprocessor(
          id,
          machine.warpSize,
          machine.sharedMemoryWords
        )
    )
  }

  get done() {
    return this.launched && this.sms.every((sm) => sm.done)
  }

  launch(program: readonly Instruction[], warpCount: number) {
    if (this.launched) throw new Error('GPU has already launched a program')
    if (program.length === 0) throw new Error('program cannot be empty')
    checkSize('warp count', warpCount)

    for (let warpId = 0; warpId < warpCount; warpId += 1) {
      const warp = new Warp(
        warpId,
        program,
        this.machine.warpSize,
        this.machine.registersPerLane
      )
      this.sms[warpId % this.sms.length].addWarp(warp)
    }

    this.launched = true
  }

  step() {
    if (!this.launched) throw new Error('launch a program before stepping')
    if (this.done) return []

    const events = this.sms.flatMap((sm) => {
      const event = sm.tick(this.cycle, this.l2)
      return event ? [event] : []
    })
    this.cycle += 1
    return events
  }

  run(maxCycles = 10_000) {
    const trace: TraceEvent[] = []

    while (!this.done) {
      if (this.cycle >= maxCycles) {
        throw new Error(`GPU did not halt within ${maxCycles} cycles`)
      }
      trace.push(...this.step())
    }

    return { cycles: this.cycle, trace, snapshot: this.snapshot() }
  }

  snapshot() {
    return {
      cycle: this.cycle,
      hbm: this.hbm.snapshot(),
      l2: this.l2.snapshot(),
      sms: this.sms.map((sm) => sm.snapshot()),
    }
  }
}

export const helloProgram: readonly Instruction[] = [
  { op: 'lane_id', destination: 0 },
  { op: 'warp_id', destination: 1 },
  { op: 'multiply_immediate', destination: 1, source: 1, value: 4 },
  { op: 'add', destination: 2, left: 1, right: 0 },
  { op: 'store_global', address: 2, source: 2 },
  { op: 'load_global', destination: 3, address: 2 },
  { op: 'store_shared', address: 0, source: 3 },
  { op: 'exit' },
]

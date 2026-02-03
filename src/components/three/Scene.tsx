'use client'

import { Canvas } from '@react-three/fiber'
import { OrbitControls, Environment } from '@react-three/drei'
import { ReactNode, Suspense } from 'react'

interface SceneProps {
  children?: ReactNode
  className?: string
}

function Fallback() {
  return (
    <mesh>
      <boxGeometry args={[1, 1, 1]} />
      <meshStandardMaterial color="hotpink" />
    </mesh>
  )
}

export default function Scene({ children, className }: SceneProps) {
  return (
    <div className={className || 'h-[400px] w-full rounded-lg border bg-muted/50'}>
      <Canvas
        camera={{ position: [0, 0, 5], fov: 50 }}
        frameloop="demand"
      >
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} />
        <Suspense fallback={<Fallback />}>
          {children || <Fallback />}
        </Suspense>
        <OrbitControls />
        <Environment preset="studio" />
      </Canvas>
    </div>
  )
}

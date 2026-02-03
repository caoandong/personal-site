'use client'

import { useEffect, useRef, type ElementType, type ReactNode } from 'react'
import { cn } from '@/lib/utils'

interface StaggerProps {
  children: ReactNode
  className?: string
  as?: ElementType
}

export function Stagger({ children, className, as: Component = 'div' }: StaggerProps) {
  const ref = useRef<HTMLElement>(null)

  useEffect(() => {
    if (!ref.current) return

    const elements = ref.current.children
    Array.from(elements).forEach((child, index) => {
      ;(child as HTMLElement).style.setProperty('--stagger', String(index))
    })
  }, [])

  return (
    <Component
      ref={ref}
      data-animate-stagger=""
      className={cn(className)}
    >
      {children}
    </Component>
  )
}

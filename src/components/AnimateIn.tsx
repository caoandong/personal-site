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

    const elements = Array.from(ref.current.children) as HTMLElement[]
    let staggerIndex = 0

    elements.forEach((child) => {
      const rect = child.getBoundingClientRect()

      // Element is above viewport (already scrolled past) - show instantly
      if (rect.bottom <= 0) {
        child.style.animation = 'none'
        return
      }

      // Element is in or below viewport - apply stagger
      child.style.setProperty('--stagger', String(staggerIndex))
      staggerIndex++
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

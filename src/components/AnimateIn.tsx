'use client'

import { useLayoutEffect, useEffect, useRef, type ElementType, type ReactNode } from 'react'
import { cn } from '@/lib/utils'

// Use useLayoutEffect on client (runs before paint), useEffect on server (avoids SSR warning)
const useIsomorphicLayoutEffect = typeof window !== 'undefined' ? useLayoutEffect : useEffect

interface StaggerProps {
  children: ReactNode
  className?: string
  as?: ElementType
}

export function Stagger({ children, className, as: Component = 'div' }: StaggerProps) {
  const ref = useRef<HTMLElement>(null)

  useIsomorphicLayoutEffect(() => {
    if (!ref.current) return

    const elements = Array.from(ref.current.children) as HTMLElement[]
    let staggerIndex = 0

    elements.forEach((child) => {
      const rect = child.getBoundingClientRect()

      // Element is above viewport (already scrolled past) - show instantly without animation
      if (rect.bottom <= 0) {
        child.style.animation = 'none'
        // Reset to visible state since CSS sets initial opacity: 0
        child.style.opacity = '1'
        child.style.filter = 'none'
        child.style.transform = 'none'
        return
      }

      // Element is in or below viewport - apply stagger delay
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

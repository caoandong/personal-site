import { cn } from '@/lib/utils'
import { type ElementType, type ReactNode } from 'react'

interface StaggerProps {
  children: ReactNode
  className?: string
  as?: ElementType
}

export function Stagger({ children, className, as: Component = 'div' }: StaggerProps) {
  return (
    <Component data-animate-stagger="" className={cn(className)}>
      {children}
    </Component>
  )
}

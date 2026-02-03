import { cn } from '@/lib/utils'
import { type ElementType, type ReactNode, type CSSProperties } from 'react'

interface AnimateInProps {
  children: ReactNode
  stagger?: number
  className?: string
  as?: ElementType
}

export function AnimateIn({
  children,
  stagger = 0,
  className,
  as: Component = 'div',
}: AnimateInProps) {
  return (
    <Component
      data-animate=""
      style={{ '--stagger': stagger } as CSSProperties}
      className={cn(className)}
    >
      {children}
    </Component>
  )
}

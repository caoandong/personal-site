'use client'

import { cn } from '@/lib/utils'
import { typography } from '@/lib/typography'
import { hoverTextColors, colorTransitions } from '@/lib/colors'
import { useNavigation } from './NavigationProvider'

interface BackButtonProps {
  fallback?: string
  className?: string
  children?: React.ReactNode
}

export function BackButton({ fallback = '/', className, children }: BackButtonProps) {
  const { goBack } = useNavigation()

  return (
    <button
      onClick={() => goBack(fallback)}
      className={cn(
        typography({ variant: 'nav', color: 'muted' }),
        hoverTextColors.default,
        colorTransitions.default,
        'cursor-pointer',
        className
      )}
    >
      {children ?? <>&larr; Back</>}
    </button>
  )
}

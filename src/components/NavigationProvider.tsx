'use client'

import { createContext, useContext, useEffect, useCallback, useMemo, useSyncExternalStore } from 'react'
import { usePathname } from 'next/navigation'
import { useRouter } from 'next/navigation'

// TypeScript interface for the Navigation API (experimental browser feature)
interface Navigation extends EventTarget {
  canGoBack: boolean
  canGoForward: boolean
  back(): void
  forward(): void
}

declare global {
  interface Window {
    navigation?: Navigation
  }
}

interface NavigationContextType {
  canGoBack: boolean
  goBack: (fallback?: string) => void
}

const NavigationContext = createContext<NavigationContextType>({
  canGoBack: false,
  goBack: () => {},
})

const HISTORY_KEY = 'nav_history'

function hasNavigationApi(): boolean {
  return typeof window !== 'undefined' && 'navigation' in window && window.navigation !== undefined
}

function getNavigationCanGoBack(): boolean {
  if (!hasNavigationApi()) return false
  return window.navigation!.canGoBack
}

function subscribeToNavigation(callback: () => void): () => void {
  if (!hasNavigationApi()) return () => {}
  const nav = window.navigation!
  nav.addEventListener('navigate', callback)
  return () => nav.removeEventListener('navigate', callback)
}

function parseHistoryFromStorage(): string[] {
  if (typeof window === 'undefined') return []
  try {
    const historyJson = sessionStorage.getItem(HISTORY_KEY)
    if (!historyJson) return []
    const parsed = JSON.parse(historyJson)
    if (Array.isArray(parsed) && parsed.every(item => typeof item === 'string')) {
      return parsed
    }
    return []
  } catch {
    return []
  }
}

// Session storage history listeners
const sessionStorageListeners = new Set<() => void>()

function notifySessionStorageListeners() {
  sessionStorageListeners.forEach(listener => listener())
}

function getSessionStorageCanGoBack(): boolean {
  return parseHistoryFromStorage().length > 1
}

function subscribeToSessionStorage(callback: () => void): () => void {
  sessionStorageListeners.add(callback)
  return () => sessionStorageListeners.delete(callback)
}

function updateSessionStorageHistory(pathname: string): void {
  const history = parseHistoryFromStorage()
  if (history[history.length - 1] !== pathname) {
    history.push(pathname)
    sessionStorage.setItem(HISTORY_KEY, JSON.stringify(history))
    notifySessionStorageListeners()
  }
}

export function NavigationProvider({ children }: { children: React.ReactNode }) {
  const pathname = usePathname()
  const router = useRouter()

  // Use useSyncExternalStore for Navigation API
  const navigationApiCanGoBack = useSyncExternalStore(
    subscribeToNavigation,
    getNavigationCanGoBack,
    () => false
  )

  // Use useSyncExternalStore for sessionStorage-based tracking
  const sessionStorageCanGoBack = useSyncExternalStore(
    subscribeToSessionStorage,
    getSessionStorageCanGoBack,
    () => false
  )

  // Update sessionStorage history when pathname changes (for fallback only)
  useEffect(() => {
    if (typeof window === 'undefined' || hasNavigationApi()) return
    updateSessionStorageHistory(pathname)
  }, [pathname])

  const canGoBack = hasNavigationApi() ? navigationApiCanGoBack : sessionStorageCanGoBack

  const goBack = useCallback((fallback = '/') => {
    if (typeof window === 'undefined') {
      router.push(fallback)
      return
    }

    // Try modern Navigation API first
    if (hasNavigationApi()) {
      if (window.navigation!.canGoBack) {
        router.back()
        return
      }
      router.push(fallback)
      return
    }

    // Fallback: sessionStorage-based tracking
    const history = parseHistoryFromStorage()

    if (history.length > 1) {
      // Remove current page from history
      history.pop()
      sessionStorage.setItem(HISTORY_KEY, JSON.stringify(history))
      notifySessionStorageListeners()
      router.back()
    } else {
      // No history, go to fallback
      sessionStorage.setItem(HISTORY_KEY, JSON.stringify([fallback]))
      notifySessionStorageListeners()
      router.push(fallback)
    }
  }, [router])

  const contextValue = useMemo(
    () => ({ canGoBack, goBack }),
    [canGoBack, goBack]
  )

  return (
    <NavigationContext.Provider value={contextValue}>
      {children}
    </NavigationContext.Provider>
  )
}

export function useNavigation() {
  return useContext(NavigationContext)
}

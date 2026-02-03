/**
 * LightRay - Animated decorative light effect
 *
 * Multiple overlapping rays with slightly different properties
 * to create a natural, organic light effect.
 * All configuration via CSS variables in globals.css.
 */
export function LightRay() {
  return (
    <div className="lightray-container opacity-80" aria-hidden="true">
      <div className="lightray lightray-1" />
      <div className="lightray lightray-2" />
      <div className="lightray lightray-center" />
      <div className="lightray lightray-3" />
      <div className="lightray lightray-4" />
      <div className="lightray lightray-5" />
    </div>
  )
}

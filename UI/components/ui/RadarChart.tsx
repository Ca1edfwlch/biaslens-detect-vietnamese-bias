'use client'

export type PerformanceTier = 'strong' | 'mixed' | 'weak' | 'limited-data'

export interface RadarChartDatum {
  key: string
  shortLabel: string
  fullLabel: string
  value: number // 0-1
  support: number
  tier: PerformanceTier
}

interface RadarChartProps {
  data: RadarChartDatum[]
  size?: number
}

// Reuses the app's existing status colors (tailwind.config.ts), but never
// alone -- validate_palette.js (dataviz skill) flags status.green vs
// status.amber as too close for red-green color blindness (CVD ΔE 4.0,
// below the safe floor). Each tier also gets a distinct marker SHAPE below,
// so identity never depends on hue perception alone.
const TIER_COLOR: Record<PerformanceTier, string> = {
  strong: '#6fb96f',
  mixed: '#e9a84a',
  weak: '#e06060',
  'limited-data': 'rgba(255,255,255,0.45)',
}

const TIER_LABEL: Record<PerformanceTier, string> = {
  strong: 'Strong signal',
  mixed: 'Mixed signal',
  weak: 'Weak signal',
  'limited-data': 'Limited data',
}

function trianglePoints(cx: number, cy: number, r: number): string {
  return [0, 120, 240]
    .map((deg) => {
      const rad = ((deg - 90) * Math.PI) / 180
      return `${cx + r * Math.cos(rad)},${cy + r * Math.sin(rad)}`
    })
    .join(' ')
}

function Marker({ tier, cx, cy }: { tier: PerformanceTier; cx: number; cy: number }) {
  const color = TIER_COLOR[tier]
  const r = 5
  const surfaceRing = { stroke: '#191c26', strokeWidth: 1.5 }

  switch (tier) {
    case 'strong':
      return <circle cx={cx} cy={cy} r={r} fill={color} {...surfaceRing} />
    case 'mixed':
      return <rect x={cx - r} y={cy - r} width={r * 2} height={r * 2} rx={1.5} fill={color} {...surfaceRing} />
    case 'weak':
      return <polygon points={trianglePoints(cx, cy, r + 1.5)} fill={color} {...surfaceRing} />
    case 'limited-data':
      return <circle cx={cx} cy={cy} r={r} fill="none" stroke={color} strokeWidth={2} strokeDasharray="2,2" />
  }
}

export function RadarChart({ data, size = 440 }: RadarChartProps) {
  const cx = size / 2
  const cy = size / 2
  const maxRadius = size * 0.32
  const labelRadius = maxRadius + 30
  const n = data.length

  function axisPoint(i: number, radius: number) {
    const angle = -Math.PI / 2 + (i * 2 * Math.PI) / n
    return { x: cx + radius * Math.cos(angle), y: cy + radius * Math.sin(angle), angle }
  }

  const rings = [0.25, 0.5, 0.75, 1]
  const dataPoints = data.map((d, i) => axisPoint(i, maxRadius * Math.max(d.value, 0.03)))
  const polygonPath = dataPoints.map((p) => `${p.x},${p.y}`).join(' ')

  return (
    <svg
      viewBox={`0 0 ${size} ${size}`}
      className="w-full h-auto max-w-sm mx-auto"
      role="img"
      aria-label="Detection performance by bias category"
    >
      {rings.map((r) => (
        <polygon
          key={r}
          points={data.map((_, i) => axisPoint(i, maxRadius * r)).map((p) => `${p.x},${p.y}`).join(' ')}
          fill="none"
          stroke="rgba(255,255,255,0.28)"
          strokeWidth={1}
        />
      ))}

      {data.map((d, i) => {
        const p = axisPoint(i, maxRadius)
        return <line key={d.key} x1={cx} y1={cy} x2={p.x} y2={p.y} stroke="rgba(255,255,255,0.28)" strokeWidth={1} />
      })}

      <polygon
        points={polygonPath}
        fill="#5b4fcf"
        fillOpacity={0.18}
        stroke="#5b4fcf"
        strokeWidth={2}
        strokeLinejoin="round"
      />

      {data.map((d, i) => {
        const p = dataPoints[i]
        return (
          <g key={d.key}>
            <circle cx={p.x} cy={p.y} r={11} fill="transparent">
              <title>
                {d.fullLabel}: {Math.round(d.value * 100)}% -- {TIER_LABEL[d.tier]} ({d.support} test example
                {d.support === 1 ? '' : 's'})
              </title>
            </circle>
            <Marker tier={d.tier} cx={p.x} cy={p.y} />
          </g>
        )
      })}

      {data.map((d, i) => {
        const p = axisPoint(i, labelRadius)
        const cos = Math.cos(p.angle)
        const anchor = cos > 0.15 ? 'start' : cos < -0.15 ? 'end' : 'middle'
        return (
          <text
            key={d.key}
            x={p.x}
            y={p.y}
            textAnchor={anchor}
            dominantBaseline="middle"
            fontSize={11}
            fill="rgba(255,255,255,0.6)"
          >
            {d.shortLabel}
          </text>
        )
      })}
    </svg>
  )
}

export function RadarChartLegend() {
  const tiers: PerformanceTier[] = ['strong', 'mixed', 'weak', 'limited-data']
  return (
    <div className="flex flex-wrap items-center justify-center gap-x-4 gap-y-2 text-xs text-text-3">
      {tiers.map((tier) => (
        <div key={tier} className="flex items-center gap-1.5">
          <svg width={14} height={14} viewBox="0 0 14 14" aria-hidden="true">
            <Marker tier={tier} cx={7} cy={7} />
          </svg>
          <span>{TIER_LABEL[tier]}</span>
        </div>
      ))}
    </div>
  )
}

import { ImageResponse } from 'next/og'

export const size = {
  width: 32,
  height: 32,
}
export const contentType = 'image/png'

export default function Icon() {
  return new ImageResponse(
    (
      <div
        style={{
          fontSize: 20,
          background: '#d4d4d4',
          width: '100%',
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          borderRadius: 6,
          color: '#171717',
          fontWeight: 700,
          fontFamily: 'system-ui, sans-serif',
        }}
      >
        A
      </div>
    ),
    {
      ...size,
    }
  )
}

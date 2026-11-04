import { NextResponse } from 'next/server'

const DETECTION_API_URL = process.env.DETECTION_API_URL || 'http://localhost:8000'

export async function GET() {
  let upstream: Response
  try {
    upstream = await fetch(`${DETECTION_API_URL}/metrics`, { cache: 'no-store' })
  } catch {
    return NextResponse.json(
      { detail: 'Detection service is unreachable. Is uvicorn running (see README)?' },
      { status: 502 },
    )
  }

  const data = await upstream.json()
  return NextResponse.json(data, { status: upstream.status })
}

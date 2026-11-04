import { NextRequest, NextResponse } from 'next/server'

const DETECTION_API_URL = process.env.DETECTION_API_URL || 'http://localhost:8000'

export async function POST(req: NextRequest) {
  const body = await req.json()

  let upstream: Response
  try {
    upstream = await fetch(`${DETECTION_API_URL}/detect-severity`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
  } catch {
    return NextResponse.json(
      { detail: 'Detection service is unreachable. Is uvicorn running (see README)?' },
      { status: 502 },
    )
  }

  const data = await upstream.json()
  return NextResponse.json(data, { status: upstream.status })
}

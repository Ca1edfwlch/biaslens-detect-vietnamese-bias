'use client'

import { useState } from 'react'
import { Button } from '@/components/ui'
import { Sparkles } from 'lucide-react'

interface InputStageProps {
  onSubmit: (text: string) => void
  loading: boolean
}

export function InputStage({ onSubmit, loading }: InputStageProps) {
  const [text, setText] = useState('')

  return (
    <div className="rounded-lg bg-bg-2 border border-border-subtle overflow-hidden">
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Paste AI-generated Vietnamese text to analyze..."
        rows={5}
        className="w-full resize-none bg-transparent px-4 py-4 text-[15px] leading-relaxed text-text-1 placeholder:text-text-4 outline-none"
      />

      <div className="flex items-center justify-between px-4 py-3 border-t border-border">
        <span className="text-xs text-text-4">{text.length} characters</span>
        <Button
          icon={<Sparkles size={15} />}
          disabled={!text.trim()}
          loading={loading}
          onClick={() => onSubmit(text.trim())}
        >
          Analyze text
        </Button>
      </div>
    </div>
  )
}

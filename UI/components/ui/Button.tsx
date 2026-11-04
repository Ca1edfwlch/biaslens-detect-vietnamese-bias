import React from 'react'
import clsx from 'clsx'

type ButtonVariant = 'primary' | 'secondary' | 'accent'
type ButtonSize = 'sm' | 'md' | 'lg'

const variantClasses: Record<ButtonVariant, string> = {
  primary: 'bg-accent hover:bg-accent-hover active:scale-95 text-white',
  secondary: 'bg-bg-2 border border-border-subtle hover:bg-bg-3 text-text-2 hover:text-text-1',
  accent: 'bg-accent-dim border border-accent-border hover:bg-opacity-25 text-accent-text',
}

const sizeClasses: Record<ButtonSize, string> = {
  sm: 'px-3 py-2 text-xs gap-1.5',
  md: 'px-4 py-3 text-sm gap-2',
  lg: 'px-5 py-4 text-base gap-2',
}

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  icon?: React.ReactNode
  loading?: boolean
  variant?: ButtonVariant
  size?: ButtonSize
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = 'primary', size = 'md', icon, loading, children, disabled, ...props }, ref) => (
    <button
      ref={ref}
      disabled={disabled || loading}
      className={clsx(
        'inline-flex items-center justify-center rounded-md font-medium transition-all duration-150 disabled:opacity-50 disabled:cursor-not-allowed',
        variantClasses[variant],
        sizeClasses[size],
        className,
      )}
      {...props}
    >
      {icon && <span className="flex-shrink-0">{icon}</span>}
      {children}
    </button>
  ),
)

Button.displayName = 'Button'

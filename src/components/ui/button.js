'use client'

import * as React from "react"

const Button = React.forwardRef(
  ({ className, variant = "default", size = "default", asChild = false, ...props }, ref) => {
    
    let compClass = "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0"
    
    if (variant === "default") {
        compClass += " bg-primary text-primary-foreground shadow hover:bg-primary/90"
    } else if (variant === "destructive") {
        compClass += " bg-destructive text-destructive-foreground shadow-sm hover:bg-destructive/90"
    } else if (variant === "outline") {
        compClass += " border border-input bg-background shadow-sm hover:bg-accent hover:text-accent-foreground"
    } else if (variant === "secondary") {
        compClass += " bg-secondary text-secondary-foreground shadow-sm hover:bg-secondary/80"
    } else if (variant === "ghost") {
        compClass += " hover:bg-accent hover:text-accent-foreground"
    } else if (variant === "link") {
        compClass += " text-primary underline-offset-4 hover:underline"
    }

    if (size === "default") {
        compClass += " h-9 px-4 py-2"
    } else if (size === "sm") {
        compClass += " h-8 rounded-md px-3 text-xs"
    } else if (size === "lg") {
        compClass += " h-10 rounded-md px-8"
    } else if (size === "icon") {
        compClass += " h-9 w-9"
    }
    
    const Component = asChild ? "span" : "button"
    
    return (
      <Component
        className={`${compClass} ${className || ''}`}
        ref={ref}
        {...props}
      />
    )
  }
)
Button.displayName = "Button"

export { Button }

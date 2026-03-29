import { cn } from "@/lib/utils.ts";

interface StatusBarProps {
  message: string;
  wsConnected: boolean;
}

export function StatusBar({ message, wsConnected }: StatusBarProps) {
  return (
    <div className="flex h-7 shrink-0 items-center justify-between border-t px-5 text-xs text-muted-foreground">
      <span>{message}</span>
      <span
        className={cn(
          "inline-block h-2 w-2 rounded-full",
          wsConnected ? "bg-emerald-500" : "bg-zinc-600",
        )}
        title={wsConnected ? "WebSocket connected" : "WebSocket disconnected"}
      />
    </div>
  );
}

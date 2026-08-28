import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  error: Error | null;
}

export default class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("ErrorBoundary caught:", error, info.componentStack);
  }

  render() {
    if (this.state.error) {
      return (
        this.props.fallback ?? (
          <div className="flex h-screen items-center justify-center bg-slate-950 text-slate-200">
            <div className="max-w-md rounded-xl border border-rose-500/30 bg-slate-900 p-6 text-center">
              <h2 className="mb-2 text-lg font-semibold text-rose-400">Something went wrong</h2>
              <p className="mb-4 text-sm text-slate-400">{this.state.error.message}</p>
              <button
                onClick={() => this.setState({ error: null })}
                className="rounded-lg bg-rose-600 px-4 py-2 text-sm font-medium text-white hover:bg-rose-500"
              >
                Try again
              </button>
            </div>
          </div>
        )
      );
    }
    return this.props.children;
  }
}

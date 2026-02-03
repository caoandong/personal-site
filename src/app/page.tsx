import Link from "next/link";
import { ThemeToggle } from "@/components/ThemeToggle";
import { cn } from "@/lib/utils";
import { typography, layout } from "@/lib/typography";

export default function Home() {
  return (
    <div className={layout.container}>
      <header className={cn(layout.headerEnd, layout.headerSpacing)}>
        <ThemeToggle />
      </header>

      <main className="space-y-8 md:space-y-10">
        <h1 className={cn(typography({ variant: "h1" }), "md:leading-tight")}>
          Personal Blog
        </h1>

        <p className={cn(typography({ variant: "body", color: "muted" }), "max-w-2xl")}>
          A minimalistic blog exploring mathematics, design engineering, and
          interactive visualizations.
        </p>

        <nav className="flex gap-8 pt-4">
          <Link
            href="/blog"
            className={cn(typography({ variant: "nav" }), "hover:text-foreground transition-colors duration-200")}
          >
            Read the blog &rarr;
          </Link>
        </nav>
      </main>

      <footer className={cn("mt-40 md:mt-64 pt-10 border-t border-stroke", typography({ variant: "small", color: "muted" }))}></footer>
    </div>
  );
}

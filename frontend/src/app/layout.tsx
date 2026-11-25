import type { Metadata } from "next";
import "./globals.css";
import { Suspense } from "react";
import { Sidebar } from "@/components/Sidebar";
import { Header } from "@/components/Header";

export const metadata: Metadata = {
  title: "TalentLens | Candidate Screening & Evaluation",
  description:
    "Evidence-grounded candidate evaluation and resume screening. Objective qualification matching and recruiter decision support.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Plus+Jakarta+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400;1,600&display=swap"
          rel="stylesheet"
        />
        <link
          href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200"
          rel="stylesheet"
        />
      </head>
      <body className="bg-slate-50 font-body-md text-slate-900 antialiased min-h-screen">
        <Sidebar />
        <div className="pl-72 flex flex-col min-h-screen">
          <Header />
          <main className="flex-1 pt-16">
            <Suspense
              fallback={
                <div className="py-20 text-center text-xs text-slate-400">
                  <div className="animate-spin w-8 h-8 border-2 border-slate-900 border-t-transparent rounded-full mx-auto mb-2" />
                  Loading workspace...
                </div>
              }
            >
              {children}
            </Suspense>
          </main>

          <footer className="border-t border-slate-200 bg-white py-4 px-8 text-center text-xs text-slate-500">
            <div className="max-w-6xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <span className="font-semibold text-slate-700">TalentLens Screening Platform</span>
                <span>•</span>
                <span>Evidence-based candidate evaluation</span>
              </div>
              <p className="text-[11px] text-slate-500">
                Notice: Decision support system. Final hiring determinations remain with human recruiters.
              </p>
            </div>
          </footer>
        </div>
      </body>
    </html>
  );
}

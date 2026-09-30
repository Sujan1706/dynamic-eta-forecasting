import type { Metadata } from "next";
import "./globals.css";
import Navbar from "@/components/Navbar";

export const metadata: Metadata = {
  title: "Dynamic ETA - Railway Operations Intelligence",
  description:
    "Real-time operational telemetry, baseline heuristic ETAs, and chained XGBoost forecasts for Indian Railways.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="h-full">
      <body className="min-h-full flex flex-col bg-[#F5F7F8] text-[#172026] antialiased selection:bg-blue-100 selection:text-blue-900">
        <Navbar />
        <main className="flex-1 max-w-[1560px] w-full mx-auto px-3 sm:px-4 md:px-5 lg:px-6 xl:px-8 py-4 sm:py-5">
          {children}
        </main>
        <footer className="border-t border-[#D9DEE3] bg-[#FFFFFF] py-3 text-xs text-[#8A949C]">
          <div className="max-w-[1560px] mx-auto px-3 sm:px-4 md:px-5 lg:px-6 xl:px-8 flex flex-col sm:flex-row items-center justify-between gap-2">
            <div>
              <span className="font-semibold text-[#172026]">Dynamic Train ETA Forecasting</span>
              <span className="hidden sm:inline"> &bull; Railway Operations Intelligence</span>
            </div>
            <div className="flex items-center space-x-3 text-[11px] font-mono text-[#8A949C]">
              <span>Chained XGBoost + Baseline Heuristic</span>
              <span>&bull;</span>
              <span className="text-[#168A55] font-semibold">System Operational</span>
            </div>
          </div>
        </footer>
      </body>
    </html>
  );
}

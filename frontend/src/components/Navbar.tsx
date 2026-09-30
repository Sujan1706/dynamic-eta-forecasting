"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";

export default function Navbar() {
  const pathname = usePathname();
  const [backendStatus, setBackendStatus] = useState<"checking" | "online" | "offline">("checking");
  const [clockTime, setClockTime] = useState<string>("");

  useEffect(() => {
    const updateClock = () => {
      const now = new Date();
      setClockTime(
        now.toLocaleTimeString("en-GB", {
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
          hour12: false,
        })
      );
    };
    updateClock();
    const timer = setInterval(updateClock, 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    let isMounted = true;
    async function checkHealth() {
      try {
        const res = await api.getHealth();
        if (isMounted) {
          setBackendStatus(res.status === "ok" ? "online" : "offline");
        }
      } catch {
        if (isMounted) {
          setBackendStatus("offline");
        }
      }
    }

    checkHealth();
    const interval = setInterval(checkHealth, 15000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  const navItems = [
    { href: "/", label: "Control Room", shortLabel: "Control Room" },
    { href: "/passenger", label: "Passenger Lookup", shortLabel: "Passenger" },
    { href: "/station", label: "Station Board", shortLabel: "Station" },
    { href: "/model-performance", label: "Model Evaluation", shortLabel: "Evaluation" },
  ];

  return (
    <header className="sticky top-0 z-40 bg-[#FFFFFF] border-b border-[#D9DEE3] shadow-[0_1px_2px_rgba(0,0,0,0.03)]">
      <div className="max-w-[1560px] mx-auto px-3 sm:px-4 md:px-5 lg:px-6 xl:px-8">
        <div className="flex items-center justify-between h-14 sm:h-16 gap-3">
          {/* Left: Brand & Operations Subtitle */}
          <div className="flex items-center space-x-3 shrink-0">
            <Link href="/" className="flex items-center space-x-2.5 group">
              <div className="w-8 h-8 rounded-md border border-[#D9DEE3] bg-[#F0F3F5] text-[#172026] flex items-center justify-center text-sm shadow-xs group-hover:border-[#2563A8] transition-colors">
                <svg
                  className="w-4 h-4 text-[#172026]"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <rect x="4" y="3" width="16" height="16" rx="2" />
                  <path d="M4 11h16" />
                  <path d="M12 3v8" />
                  <path d="m8 19-2 3" />
                  <path d="m16 19 2 3" />
                  <circle cx="8" cy="15" r="1" />
                  <circle cx="16" cy="15" r="1" />
                </svg>
              </div>
              <div className="leading-tight">
                <span className="font-bold text-[15px] sm:text-base text-[#172026] tracking-tight block">
                  Dynamic ETA
                </span>
                <span className="text-[9px] text-[#66717A] font-semibold tracking-wider uppercase block">
                  Railway Operations Intelligence
                </span>
              </div>
            </Link>
          </div>

          {/* Center: Segmented Control Navigation */}
          <nav className="flex items-center p-1 bg-[#F0F3F5] border border-[#D9DEE3] rounded-lg overflow-x-auto scrollbar-none">
            {navItems.map((item) => {
              const isActive =
                item.href === "/"
                  ? pathname === "/" || pathname.startsWith("/trains")
                  : pathname === item.href;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`px-3 sm:px-4 py-1.5 rounded-md text-xs sm:text-[13px] font-medium whitespace-nowrap transition-all duration-150 ${
                    isActive
                      ? "bg-[#FFFFFF] text-[#172026] font-semibold shadow-xs border border-[#D9DEE3]"
                      : "text-[#66717A] hover:text-[#172026] hover:bg-[#FFFFFF]/60 border border-transparent"
                  }`}
                >
                  <span className="sm:hidden">{item.shortLabel}</span>
                  <span className="hidden sm:inline">{item.label}</span>
                </Link>
              );
            })}
          </nav>

          {/* Right: IST Clock & Backend Telemetry */}
          <div className="flex items-center space-x-3.5 shrink-0">
            {/* Real-time Clock */}
            <div className="hidden sm:block text-right">
              <div className="text-[9px] uppercase tracking-wider text-[#8A949C] font-bold">
                IST Clock
              </div>
              <div className="text-xs font-mono font-bold text-[#172026] tracking-tight">
                {clockTime || "--:--:--"}
              </div>
            </div>

            {/* FastAPI Status Pill */}
            <div
              className={`flex items-center space-x-1.5 text-xs px-2.5 py-1 rounded-md border font-medium ${
                backendStatus === "online"
                  ? "bg-[#EBF7EE] text-[#168A55] border-[#B4E2C1]"
                  : backendStatus === "checking"
                  ? "bg-amber-50 text-[#B77900] border-amber-200"
                  : "bg-rose-50 text-[#D64545] border-rose-200"
              }`}
            >
              <span
                className={`w-1.5 h-1.5 rounded-full ${
                  backendStatus === "online"
                    ? "bg-[#168A55] animate-pulse"
                    : backendStatus === "checking"
                    ? "bg-[#B77900]"
                    : "bg-[#D64545]"
                }`}
              />
              <span className="font-mono text-[11px] font-semibold">
                FastAPI: {backendStatus.toUpperCase()}
              </span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}

import type { Metadata } from "next";
import "./globals.css";
import { Sidebar } from "@/components/Sidebar";
import { TopBar } from "@/components/TopBar";
import { MobileToggle } from "@/components/MobileToggle";
import { AuthProvider } from "@/app/lib/auth-context";

export const metadata: Metadata = {
  title: "Razorpay CloseLoop — Financial Operations",
  description:
    "Automated financial exception resolution with safety guardrails",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <AuthProvider>
          <MobileToggle />
          <div className="overlay" id="overlay" />
          <Sidebar />
          <div className="main-content">
            <TopBar />
            <div className="page-container">{children}</div>
          </div>
        </AuthProvider>
      </body>
    </html>
  );
}

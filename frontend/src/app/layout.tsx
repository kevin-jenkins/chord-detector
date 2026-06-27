import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Chord Detector",
  description: "Local-only chord detection",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

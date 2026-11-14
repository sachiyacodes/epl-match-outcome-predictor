import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Touchline | EPL Match Predictor",
  description: "Historical Premier League match probabilities and honest time-based model evaluation.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}

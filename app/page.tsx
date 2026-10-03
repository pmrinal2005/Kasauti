import { redirect } from "next/navigation";

// Landing page will be built later — the dashboard is the entry point for now.
export default function Home() {
  redirect("/dashboard");
}

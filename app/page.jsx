import HomeHero from "../components/HomeHero";
import HomeClient from "../components/HomeClient";
import QuickStart from "../components/QuickStart";
import { buildCategories } from "../lib/docs";

export default function Home() {
  const categories = buildCategories();
  const totalDocs = categories.reduce((s, c) => s + c.totalFiles, 0);
  const totalCategories = categories.length;

  return (
    <div className="home-page">
      <HomeHero totalDocs={totalDocs} totalCategories={totalCategories} />
      <QuickStart />
      <HomeClient categories={categories} />
    </div>
  );
}
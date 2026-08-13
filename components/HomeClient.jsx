"use client";

import { useState } from "react";
import CategoryCard from "./CategoryCard";
import CategoryExplorer from "./CategoryExplorer";

export default function HomeClient({ categories }) {
  const [activeCategory, setActiveCategory] = useState(null);

  return (
    <>
      <section className="home-section">
        <div className="section-head">
          <div>
            <h2 className="section-title">Browse by topic</h2>
            <p className="section-sub">Pick a subject — the full list opens below</p>
          </div>
        </div>
        <div className="category-grid">
          {categories.map((category) => (
            <CategoryCard
              key={category.key}
              category={category}
              active={activeCategory === category.key}
              onSelect={(key) => setActiveCategory(key)}
            />
          ))}
        </div>
      </section>

      <CategoryExplorer
        categories={categories}
        activeCategory={activeCategory}
        onSelect={setActiveCategory}
      />
    </>
  );
}
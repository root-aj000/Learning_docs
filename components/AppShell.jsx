"use client";

import { useState } from "react";
import Sidebar from "./Sidebar";

export default function AppShell({ docs, children }) {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <div className="app-layout">
      <Sidebar
        docs={docs}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />
      <div className="main-content">
        <header className="content-header">
          <div className="content-header-inner">
            <button
              className="mobile-menu-btn"
              onClick={() => setSidebarOpen(true)}
              aria-label="Open menu"
            >
              <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
                <line x1="3" y1="5" x2="17" y2="5" />
                <line x1="3" y1="10" x2="17" y2="10" />
                <line x1="3" y1="15" x2="17" y2="15" />
              </svg>
            </button>
            <div className="breadcrumb">
              <a href="/">Home</a>
            </div>
          </div>
        </header>
        <div className="content-body">
          {children}
        </div>
      </div>
    </div>
  );
}

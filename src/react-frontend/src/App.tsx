import { useState } from "react";
import ImageProcessing from "./pages/ImageProcessing";
import ModelConfig from "./pages/ModelConfig";
import { Images, Settings } from "lucide-react";

const tabs = [
  { id: "processing", label: "Image Processing", icon: Images },
  { id: "config", label: "Model Configuration", icon: Settings },
] as const;

type TabId = (typeof tabs)[number]["id"];

export default function App() {
  const [activeTab, setActiveTab] = useState<TabId>("processing");

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      {/* Header */}
      <header className="border-b border-gray-800 bg-gray-900/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <h1 className="text-xl font-bold tracking-tight text-white">
              Manga Translate App
            </h1>
            <nav className="flex gap-1">
              {tabs.map((tab) => {
                const Icon = tab.icon;
                const active = activeTab === tab.id;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                      active
                        ? "bg-indigo-600 text-white"
                        : "text-gray-400 hover:text-white hover:bg-gray-800"
                    }`}
                  >
                    <Icon size={16} />
                    {tab.label}
                  </button>
                );
              })}
            </nav>
          </div>
        </div>
      </header>

      {/* Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {activeTab === "processing" && <ImageProcessing />}
        {activeTab === "config" && <ModelConfig />}
      </main>
    </div>
  );
}

const sidebars = {
  mainSidebar: [
    "intro",
    {
      type: "category",
      label: "Getting Started",
      link: {
        type: "doc",
        id: "getting-started",
      },
      items: ["getting-started-gpu", "getting-started-cpu-only"],
    },
    {
      type: "category",
      label: "Run a Model",
      link: {
        type: "doc",
        id: "run-model",
      },
      items: ["run-model-cli", "run-model-webui"],
    },
    "cli-reference",
    "web-ui",
    "public-beta",
  ],
};

module.exports = sidebars;

// @ts-check

const config = {
  title: "vAquila Docs",
  tagline: "Open-source LLM orchestration with vLLM + Docker",
  favicon: "img/logo-base.png",

  url: "https://xschahl.github.io",
  baseUrl: "/vAquila/",

  organizationName: "xschahl",
  projectName: "vAquila",

  onBrokenLinks: "throw",
  markdown: {
    hooks: {
      onBrokenMarkdownLinks: "warn",
    },
  },

  i18n: {
    defaultLocale: "en",
    locales: ["en"],
  },

  presets: [
    [
      "classic",
      {
        docs: {
          sidebarPath: require.resolve("./sidebars.js"),
          routeBasePath: "/",
        },
        blog: false,
        theme: {
          customCss: require.resolve("./src/css/custom.css"),
        },
      },
    ],
  ],

  themeConfig: {
    image: "img/logo-base.png",
    navbar: {
      title: "vAquila",
      logo: {
        alt: "vAquila logo",
        src: "img/logo-base.png",
      },
      items: [
        {
          type: "docSidebar",
          sidebarId: "mainSidebar",
          position: "left",
          label: "Documentation",
        },
        {
          href: "https://github.com/xschahl/vAquila",
          label: "GitHub",
          position: "right",
        },
      ],
    },
    footer: {
      style: "dark",
      links: [
        {
          title: "Docs",
          items: [
            {
              label: "Getting Started",
              to: "/getting-started",
            },
            {
              label: "CLI Reference",
              to: "/cli-reference",
            },
          ],
        },
        {
          title: "Community",
          items: [
            {
              label: "Issues",
              href: "https://github.com/xschahl/vAquila/issues",
            },
          ],
        },
      ],
      copyright: `Copyright ${new Date().getFullYear()} vAquila contributors`,
    },
    prism: {
      additionalLanguages: ["bash"],
    },
  },
};

module.exports = config;

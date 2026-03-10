import React, { useEffect, useState } from 'react';
import Head from '@docusaurus/Head';
import Link from '@docusaurus/Link';
import styles from './index.module.css';

function CustomNavbar() {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 50);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <nav className={`${styles.navbar} ${scrolled ? styles.navbarScrolled : ''}`}>
      <div className={styles.navbarInner}>
        <div className={styles.navbarBrand}>
          <img src="/vAquila/img/logo-base.png" alt="vAquila Logo" className={styles.navbarLogo} />
          <span className={styles.navbarTitle}>vAquila</span>
        </div>
        <div className={styles.navbarLinks}>
          <Link to="/docs/getting-started" className={styles.navLink}>Documentation</Link>
          <Link href="https://github.com/xschahl/vAquila" className={`${styles.navLink} ${styles.navLinkGithub}`}>
            <svg height="24" viewBox="0 0 16 16" version="1.1" width="24" aria-hidden="true" fill="currentColor">
              <path fillRule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path>
            </svg>
            <span>GitHub</span>
          </Link>
        </div>
      </div>
    </nav>
  );
}

function HeroSection() {
  return (
    <section className={styles.heroSection}>
      <div className={styles.heroGlow}></div>
      <div className={styles.heroContent}>
        <div className={styles.logoWrapper}>
          <img src="/vAquila/img/logo-base.png" alt="vAquila Icon" className={styles.heroMainLogo} />
        </div>
        <h1 className={styles.heroTitle}>vAquila Control Center</h1>
        <p className={styles.heroSubtitle}>
          Operate your local vLLM runtime with confidence. Launch models, monitor VRAM, 
          inspect logs, and validate inference from one reliable control surface.
        </p>
        <div className={styles.heroAction}>
          <Link to="/docs/getting-started" className={styles.ctaButton}>
            Get Started
            <span className={styles.ctaGlow}></span>
          </Link>
          <div className={styles.statusBadge}>
            <span className={styles.statusDot}></span>
            FastAPI local UI for Docker + vLLM workflows
          </div>
        </div>
      </div>
    </section>
  );
}

function DashboardPreview() {
  return (
    <section className={styles.previewSection}>
      <div className={styles.previewContainer}>
        <div className={styles.previewHeader}>
          <div className={styles.windowControls}>
            <span />
            <span />
            <span />
          </div>
          <div className={styles.windowTitle}>vAquila Dashboard Runtime</div>
        </div>
        <div className={styles.metricsGrid}>
          <div className={styles.metricCard}>
            <p className={styles.metricLabel}>Managed containers</p>
            <h2 className={styles.metricValue}>1</h2>
            <p className={styles.metricDesc}>All vAquila containers</p>
            <div className={styles.cardGlow1}></div>
          </div>
          <div className={styles.metricCard}>
            <p className={styles.metricLabel}>Running now</p>
            <h2 className={styles.metricValue}>1</h2>
            <p className={styles.metricDesc}>Currently serving requests</p>
            <div className={styles.cardGlow2}></div>
          </div>
          <div className={styles.metricCard}>
            <p className={styles.metricLabel}>Cached models</p>
            <h2 className={styles.metricValue}>2</h2>
            <p className={styles.metricDesc}>Ready from local HF cache</p>
            <div className={styles.cardGlow3}></div>
          </div>
        </div>
        
        <div className={styles.capacityCard}>
          <div className={styles.capacityHeader}>
            <span className={styles.capacityDot}></span>
            CAPACITY OVERVIEW
          </div>
          <h3 className={styles.capacityTitle}>GPU utilization</h3>
          <div className={styles.gpuStats}>
            <span>GPU 0 • NVIDIA GeForce RTX 5070 Ti Laptop GPU</span>
            <span className={styles.gpuBadge}>75.6% used</span>
          </div>
          <div className={styles.progressBarWrapper}>
            <div className={styles.progressBar} style={{width: '75.6%'}}></div>
          </div>
          <div className={styles.modelRunning}>
            <span>Qwen/Qwen3-4B-Instruct-2507-FP8</span>
            <span>5.37 GiB</span>
          </div>
        </div>
      </div>
    </section>
  );
}

function FeaturesSection() {
  return (
    <section className={styles.featuresSection}>
      <div className={styles.featureGrid}>
        <div className={styles.featureBox}>
          <div className={styles.featureIcon}>⚡</div>
          <h3>Deployment</h3>
          <p>Launch a vLLM container with explicit runtime knobs. Configure ports, context lengths, timeouts, and more.</p>
        </div>
        <div className={styles.featureBox}>
          <div className={styles.featureIcon}>🧬</div>
          <h3>Background Jobs</h3>
          <p>Track async launches and deeply inspect both task initialization logs and raw container outputs in real-time.</p>
        </div>
        <div className={styles.featureBox}>
          <div className={styles.featureIcon}>📊</div>
          <h3>Host Metrics</h3>
          <p>View detailed breakdown of CPU usage, RAM allocations, and logical core distributions per model.</p>
        </div>
      </div>
    </section>
  );
}

function EnterpriseSection() {
  return (
    <section className={styles.enterpriseSection}>
      <div className={styles.enterpriseContainer}>
        <div className={styles.enterpriseHeader}>
          <span className={styles.enterpriseBadge}>COMING SOON</span>
          <h2 className={styles.enterpriseTitle}>vAquila Enterprise</h2>
          <p className={styles.enterpriseSubtitle}>
            Scale your local AI infrastructure across teams. 
            Advanced security, compliance, and orchestration built for production.
          </p>
        </div>
        
        <div className={styles.enterpriseFeatures}>
          <div className={styles.entFeature}>
            <div className={styles.entIcon}>🛡️</div>
            <h4>SSO & SAML</h4>
            <p>Secure authentication integrating directly with your corporate identity providers.</p>
          </div>
          <div className={styles.entFeature}>
            <div className={styles.entIcon}>🔑</div>
            <h4>Role-Base Access (RBAC)</h4>
            <p>Granular permissions: control who can launch, view, or stop specific models.</p>
          </div>
          <div className={styles.entFeature}>
            <div className={styles.entIcon}>🌐</div>
            <h4>Multi-Node Clusters</h4>
            <p>Deploy and orchestrate vLLM instances across multiple remote GPU servers simultaneously.</p>
          </div>
        </div>
        
        <div className={styles.enterpriseAction}>
          <a href="mailto:xavier1.schahl@epitech.eu" className={styles.entButton}>
            Join the Waitlist
          </a>
        </div>
      </div>
    </section>
  );
}

export default function Home() {
  return (
    <>
      <Head>
        <title>vAquila | Local AI Orchestration</title>
        <meta name="description" content="Operate your local vLLM runtime with confidence via an ultra-modern control center." />
        <style>
          {`
            /* Prevent Docusaurus layout from showing the skip to content link when there is no layout */
            .skipToContent_node_modules-\\@docusaurus-theme-classic-lib-theme-SkipToContent-styles-module {
              display: none !important;
            }
          `}
        </style>
      </Head>

      <div className={styles.landingContainer}>
        <CustomNavbar />
        <main className={styles.mainContent}>
          <HeroSection />
          <DashboardPreview />
          <FeaturesSection />
          <EnterpriseSection />
        </main>
        
        <footer className={styles.customFooter}>
          <div className={styles.footerInner}>
            <p>© {new Date().getFullYear()} vAquila Contributors. Open Source under Apache 2.0 License.</p>
            <div className={styles.footerLinks}>
              <Link to="/docs/getting-started">Documentation</Link>
              <Link to="/docs/cli-reference">CLI Reference</Link>
              <Link href="https://github.com/xschahl/vAquila">GitHub Repository</Link>
            </div>
          </div>
        </footer>
      </div>
    </>
  );
}

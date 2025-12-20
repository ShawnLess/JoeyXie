// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-strong-joey-strong-xie",
    title: "<strong>Joey </strong> Xie",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-teaching",
          title: "teaching",
          description: "Materials for courses you taught. Replace this text with your description.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/teaching/";
          },
        },{id: "post-modern-c-features",
      
        title: "Modern C++ features",
      
      description: "Brief notes of modern C++ features",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/2025/06/11/Modern-C++-Features.html";
        
      },
    },{id: "post-addressing-of-gpgpu",
      
        title: "Addressing of GPGPU",
      
      description: "Addressing scheme of GPU.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/2025/03/24/Addressing-of-GPU-Scaling-Out-Up.html";
        
      },
    },{id: "post-discriminative-learning-vs-generative-learning",
      
        title: "Discriminative learning vs Generative learning",
      
      description: "A concrete comparison of the discriminative and generative learning.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/2025/02/21/Discriminative-Generative.html";
        
      },
    },{id: "post-fundamental-math-for-machine-learning",
      
        title: "Fundamental Math for Machine Learning",
      
      description: "A learning notes of the basic math knowledge and conceptions for machine learning.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/2025/01/27/Fundamental-Math-for-Machine-Learning.html";
        
      },
    },{id: "news-moved-to-united-state-learned-how-to-play-chess",
          title: 'Moved to  United State, Learned how to play chess!',
          description: "",
          section: "News",},{id: "news-来到美国-接触国际象棋",
          title: '来到美国, 接触国际象棋!',
          description: "",
          section: "News",},{id: "news-moved-to-beijing-start-professional-chess-training",
          title: 'Moved to Beijing, start professional chess training',
          description: "",
          section: "News",},{id: "news-回到北京-开始国际象棋训练",
          title: '回到北京, 开始国际象棋训练',
          description: "",
          section: "News",},{id: "news-moved-back-to-ca-united-states",
          title: 'Moved back to CA, United States.',
          description: "",
          section: "News",},{id: "news-全家重新搬到美国",
          title: '全家重新搬到美国',
          description: "",
          section: "News",},{id: "projects-mapu",
          title: 'MaPU',
          description: "A Mathematical Computing Architecture with VLIW and CGRA features.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/1_mapu/";
            },},{id: "projects-celerity",
          title: 'Celerity',
          description: "Open-Source RISC-V Tiered Accelerator Fabric SoC.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/2_celerity/";
            },},{id: "projects-sdh",
          title: 'SDH',
          description: "Software Defined Hardware.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/3_SDH/";
            },},{id: "projects-yitian710",
          title: 'YiTian710',
          description: "Alibaba’s Yitian 710 Is China’s First Homegrown Cloud-Native CPU to Be Put Into Large-Scale Use,",
          section: "Projects",handler: () => {
              window.location.href = "/projects/4_YiTian710/";
            },},{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];

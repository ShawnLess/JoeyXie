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
  },{id: "nav-chessground",
          title: "ChessGround",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "nav-chesscourse",
          title: "ChessCourse",
          description: "From Pawn to King",
          section: "Navigation",
          handler: () => {
            window.location.href = "/chesscourse/";
          },
        },{id: "post-test",
      
        title: "Test",
      
      description: "",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/2025/12/25/test.html";
        
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
          section: "News",},{
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

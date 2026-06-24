/* infy docs — navigation, search, code copy */
document.addEventListener('DOMContentLoaded',()=>{
  // Active sidebar link
  const path=location.pathname.split('/').pop()||'index.html';
  document.querySelectorAll('.sidebar-link').forEach(a=>{
    const href=a.getAttribute('href');
    if(href===path||href==='./'+path)a.classList.add('active');
  });

  // Code copy buttons
  document.querySelectorAll('pre').forEach(pre=>{
    const btn=document.createElement('button');
    btn.textContent='Copy';
    btn.className='copy-btn';
    btn.style.cssText='position:absolute;top:8px;right:8px;background:#333;color:#fff;border:none;padding:4px 10px;border-radius:4px;font-size:11px;cursor:pointer;opacity:0;transition:opacity .15s;font-family:var(--mono)';
    pre.style.position='relative';
    pre.appendChild(btn);
    pre.addEventListener('mouseenter',()=>btn.style.opacity='1');
    pre.addEventListener('mouseleave',()=>btn.style.opacity='0');
    btn.addEventListener('click',()=>{
      const code=pre.querySelector('code')?.textContent||pre.textContent;
      navigator.clipboard.writeText(code).then(()=>{btn.textContent='Copied!';setTimeout(()=>btn.textContent='Copy',1500)});
    });
  });

  // Sidebar toggle on mobile
  const header=document.querySelector('.header');
  if(header&&!document.querySelector('.menu-btn')){
    const menuBtn=document.createElement('button');
    menuBtn.className='menu-btn';
    menuBtn.innerHTML='☰';
    menuBtn.style.cssText='display:none;background:none;border:none;color:#fff;font-size:20px;cursor:pointer;margin-right:12px';
    menuBtn.addEventListener('click',()=>{
      const sb=document.querySelector('.sidebar');
      sb.style.display=sb.style.display==='block'?'none':'block';
    });
    if(window.innerWidth<=768)menuBtn.style.display='block';
    header.prepend(menuBtn);
    window.addEventListener('resize',()=>{menuBtn.style.display=window.innerWidth<=768?'block':'none'});
  }

  // Smooth scroll for anchor links
  document.querySelectorAll('a[href^="#"]').forEach(a=>{
    a.addEventListener('click',e=>{
      const id=a.getAttribute('href').slice(1);
      const el=document.getElementById(id);
      if(el){e.preventDefault();el.scrollIntoView({behavior:'smooth',block:'start'})}
    });
  });
});

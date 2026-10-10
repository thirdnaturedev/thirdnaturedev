from __future__ import annotations
import io, json, math, os, random, urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw, ImageFont
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

USER = "thirdnaturedev"
W, H = 1180, 610
BG_DARK = "#0A101F"
BG_LIGHT = "#F6F7FB"
PORTRAIT_DARK = "#A78BFA"
PORTRAIT_LIGHT = "#7C3AED"
CHROME_DARK = "#22D3EE"
CHROME_LIGHT = "#0891B2"
ACCENT = "#10B981"
MUTED_DARK = "#94A3B8"
MUTED_LIGHT = "#475569"

random.seed(13)
np.random.seed(13)


def fetch_avatar() -> Image.Image:
    req = urllib.request.Request(f"https://api.github.com/users/{USER}", headers={"User-Agent":"thirdnature-profile-builder"})
    with urllib.request.urlopen(req, timeout=30) as r:
        profile = json.loads(r.read().decode())
    req = urllib.request.Request(profile["avatar_url"] + "&s=1200", headers={"User-Agent":"thirdnature-profile-builder"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return Image.open(io.BytesIO(r.read())).convert("RGB")


def cover_crop(img: Image.Image, size=(300,340)) -> Image.Image:
    # head + shoulders: keep vertical room and center slightly high
    w,h = img.size
    target = size[0]/size[1]
    if w/h > target:
        nw = int(h*target)
        left = max(0, (w-nw)//2)
        img = img.crop((left,0,left+nw,h))
    else:
        nh = int(w/target)
        top = max(0, int((h-nh)*0.36))
        img = img.crop((0,top,w,top+nh))
    return img.resize(size, Image.Resampling.LANCZOS)


def fg_mask(rgb: np.ndarray) -> np.ndarray:
    # Estimate background from border pixels, then keep largest central foreground component.
    edge = np.concatenate([rgb[:10].reshape(-1,3), rgb[-10:].reshape(-1,3), rgb[:, :10].reshape(-1,3), rgb[:, -10:].reshape(-1,3)])
    bg = np.median(edge.astype(np.float32), axis=0)
    dist = np.linalg.norm(rgb.astype(np.float32)-bg, axis=2)
    thr = max(24.0, float(np.percentile(dist, 46)))
    m = dist > thr
    m = ndimage.binary_closing(m, iterations=3)
    m = ndimage.binary_opening(m, iterations=1)
    m = ndimage.binary_fill_holes(m)
    lab,n = ndimage.label(m)
    if n:
        cy,cx = np.array(m.shape)//2
        center_lab = lab[cy,cx]
        if center_lab:
            m = lab == center_lab
        else:
            counts = np.bincount(lab.ravel()); counts[0]=0
            m = lab == counts.argmax()
    return ndimage.binary_dilation(m, iterations=1)


def floyd_steinberg(gray: np.ndarray) -> np.ndarray:
    a = gray.astype(np.float32).copy()
    h,w = a.shape
    out = np.zeros((h,w), np.uint8)
    for y in range(h):
        rtl = y % 2 == 1
        xs = range(w-1,-1,-1) if rtl else range(w)
        for x in xs:
            old = a[y,x]
            new = 255.0 if old >= 128 else 0.0
            out[y,x] = 1 if new < 128 else 0
            err = old-new
            if not rtl:
                if x+1<w: a[y,x+1] += err*7/16
                if y+1<h:
                    if x>0: a[y+1,x-1] += err*3/16
                    a[y+1,x] += err*5/16
                    if x+1<w: a[y+1,x+1] += err*1/16
            else:
                if x-1>=0: a[y,x-1] += err*7/16
                if y+1<h:
                    if x+1<w: a[y+1,x+1] += err*3/16
                    a[y+1,x] += err*5/16
                    if x-1>=0: a[y+1,x-1] += err*1/16
    return out


def portrait_bits(img: Image.Image, dark: bool) -> np.ndarray:
    img = cover_crop(img)
    img = ImageOps.autocontrast(img, cutoff=1)
    img = ImageEnhance.Contrast(img).enhance(1.3)
    img = img.filter(ImageFilter.UnsharpMask(radius=3, percent=140, threshold=2))
    rgb = np.array(img)
    g = np.array(ImageOps.grayscale(img), dtype=np.float32)
    if dark:
        mask = fg_mask(rgb)
        # Lit subject on dark panel: invert luminance, then clear diffusion bleed outside mask.
        d = floyd_steinberg(255-g)
        d[~mask] = 0
    else:
        # Light mode keeps backdrop; dark parts become ink.
        d = floyd_steinberg(g)
    return d


def row_runs(bits: np.ndarray):
    runs=[]
    h,w = bits.shape
    for y in range(h):
        x=0
        while x<w:
            if bits[y,x]:
                x0=x
                while x<w and bits[y,x]: x+=1
                runs.append((x0,y,x-x0))
            else: x+=1
    return runs


def render_mark(text: str, n=780):
    im = Image.new("L", (220,135), 0)
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 72)
    except Exception:
        font = ImageFont.load_default()
    box = dr.textbbox((0,0), text, font=font)
    tw,th=box[2]-box[0], box[3]-box[1]
    dr.text(((220-tw)/2,(135-th)/2-box[1]), text, font=font, fill=255)
    arr=np.array(im)>90
    ys,xs=np.where(arr)
    if len(xs)<n:
        idx=np.random.choice(len(xs),n,replace=True)
    else:
        idx=np.random.choice(len(xs),n,replace=False)
    pts=np.column_stack([xs[idx],ys[idx]]).astype(float)
    pts[:,0]=520 + pts[:,0]*1.55
    pts[:,1]=170 + pts[:,1]*1.45
    return pts


def match(a,b):
    # Optimal transport / minimum total Euclidean travel.
    cost=((a[:,None,:]-b[None,:,:])**2).sum(2)
    r,c=linear_sum_assignment(cost)
    out=np.empty_like(b)
    out[r]=b[c]
    return out


def esc(s:str)->str:
    return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")


def path_groups(runs, groups=94):
    buckets=[[] for _ in range(groups)]
    for x,y,l in runs:
        drift = (x/300.0)*groups + np.random.normal(0,4.0)
        gi=int(drift)%groups
        buckets[gi].append((x,y,l))
    return buckets


def intro_groups(runs, groups=60):
    b=[[] for _ in range(groups)]
    order=np.random.permutation(len(runs))
    for i,idx in enumerate(order): b[i%groups].append(runs[idx])
    return b


def path_d(items):
    # Short horizontal segments; crispEdges preserves sub-2px clarity better than glyphs.
    return "".join(f"M{x} {y}h{max(0.72,l)}" for x,y,l in items)


def svg(theme:str, bits:np.ndarray, marks):
    dark=theme=="dark"
    bg=BG_DARK if dark else BG_LIGHT
    panel="#0F172A" if dark else "#FFFFFF"
    panel2="#111827" if dark else "#EEF2F7"
    border="#243244" if dark else "#CBD5E1"
    text="#F8FAFC" if dark else "#0F172A"
    muted=MUTED_DARK if dark else MUTED_LIGHT
    chrome=CHROME_DARK if dark else CHROME_LIGHT
    portrait=PORTRAIT_DARK if dark else PORTRAIT_LIGHT
    runs=row_runs(bits)
    bands=path_groups(runs)
    intros=intro_groups(runs)
    p1,p2,p3=marks
    n=len(p1)
    key="0;0.2113;0.3028;0.4437;0.5352;0.6761;0.7676;0.9085;1"
    lines=[]
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
    lines.append('<defs><filter id="glow"><feGaussianBlur stdDeviation="3" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter><clipPath id="portraitClip"><rect x="58" y="166" width="392" height="310" rx="10"/></clipPath></defs>')
    lines.append(f'<rect width="1180" height="610" rx="24" fill="{bg}"/><rect x="16" y="16" width="1148" height="578" rx="18" fill="{panel}" stroke="{border}" stroke-width="2"/>')
    lines.append('<circle cx="42" cy="42" r="6" fill="#FF5F57"/><circle cx="62" cy="42" r="6" fill="#FEBC2E"/><circle cx="82" cy="42" r="6" fill="#28C840"/>')
    lines.append(f'<text x="110" y="47" fill="{muted}" font-size="13" font-family="ui-monospace,Menlo,Consolas,monospace">profile.sh --live</text>')
    lines.append(f'<rect x="892" y="28" width="145" height="28" rx="14" fill="{chrome}" opacity=".12"/><text x="964" y="47" text-anchor="middle" fill="{chrome}" font-size="13" font-family="ui-monospace,Menlo,Consolas,monospace">@thirdnaturedev</text>')
    lines.append(f'<rect x="1048" y="28" width="82" height="28" rx="14" fill="{panel2}" stroke="{border}"/><circle cx="1064" cy="42" r="4" fill="#EF4444"><animate attributeName="opacity" values="1;.2;1" dur="1.2s" repeatCount="indefinite"/></circle><text x="1077" y="46" fill="{text}" font-size="11" font-family="ui-monospace,Menlo,Consolas,monospace">LIVE</text>')
    lines.append(f'<text x="48" y="93" fill="{chrome}" font-size="13" font-weight="700" font-family="ui-monospace,Menlo,Consolas,monospace">VISUAL.MAP</text>')
    lines.append(f'<rect x="40" y="108" width="430" height="442" rx="14" fill="{panel2}" stroke="{border}"/><rect x="54" y="122" width="402" height="36" rx="9" fill="{panel}" stroke="{border}"/><text x="72" y="145" fill="{muted}" font-size="11" font-family="ui-monospace,Menlo,Consolas,monospace">portrait / dissolve / morph / return</text>')
    lines.append(f'<rect x="58" y="166" width="392" height="310" rx="10" fill="{bg}"/>')
    # Intro shimmer layer: random groups distributed across full portrait, no wipe.
    lines.append('<g clip-path="url(#portraitClip)" transform="translate(92 170) scale(1.05 .85)">')
    for i,g in enumerate(intros):
        if not g: continue
        delay=(i%20)*0.045 + (i//20)*0.018
        lines.append(f'<path d="{path_d(g)}" fill="none" stroke="{portrait}" stroke-width="1" shape-rendering="crispEdges" opacity="0"><animate attributeName="opacity" values="0;1" begin="{delay:.3f}s" dur="2s" fill="freeze"/></path>')
    lines.append('</g>')
    # Loop portrait bands. Add band-specific noisy drift to avoid square-grid dissolve.
    logo_cx=float(p1[:,0].mean()); logo_cy=float(p1[:,1].mean())
    lines.append('<g clip-path="url(#portraitClip)" transform="translate(92 170) scale(1.05 .85)">')
    for i,g in enumerate(bands):
        if not g: continue
        gx=np.mean([x+l/2 for x,y,l in g]); gy=np.mean([y for x,y,l in g])
        # Translate about 42% toward the first logo centroid, converted back to portrait coordinates.
        target_x=(logo_cx-92)/1.05; target_y=(logo_cy-170)/.85
        dx=(target_x-gx)*.42 + np.random.normal(0,4)
        dy=(target_y-gy)*.42 + np.random.normal(0,4)
        lines.append(f'<g opacity="0"><path d="{path_d(g)}" fill="none" stroke="{portrait}" stroke-width="1" shape-rendering="crispEdges"/><animate attributeName="opacity" begin="3.2s" dur="14.2s" repeatCount="indefinite" keyTimes="{key}" values="1;1;0;0;0;0;0;0;1"/><animateTransform attributeName="transform" type="translate" begin="3.2s" dur="14.2s" repeatCount="indefinite" keyTimes="{key}" values="0 0;0 0;{dx:.2f} {dy:.2f};{dx:.2f} {dy:.2f};{dx:.2f} {dy:.2f};{dx:.2f} {dy:.2f};{dx:.2f} {dy:.2f};{dx:.2f} {dy:.2f};0 0"/></g>')
    lines.append('</g>')
    # Travellers morph among TN, </>, >_. Hidden during portrait phase.
    for i in range(n):
        a,b,c=p1[i],p2[i],p3[i]
        xs=f'{a[0]:.1f};{a[0]:.1f};{a[0]:.1f};{a[0]:.1f};{a[0]:.1f};{b[0]:.1f};{b[0]:.1f};{c[0]:.1f};{c[0]:.1f}'
        ys=f'{a[1]:.1f};{a[1]:.1f};{a[1]:.1f};{a[1]:.1f};{a[1]:.1f};{b[1]:.1f};{b[1]:.1f};{c[1]:.1f};{c[1]:.1f}'
        lines.append(f'<circle cx="{a[0]:.1f}" cy="{a[1]:.1f}" r="1.15" fill="{chrome}" opacity="0"><animate attributeName="cx" begin="3.2s" dur="14.2s" repeatCount="indefinite" keyTimes="{key}" values="{xs}"/><animate attributeName="cy" begin="3.2s" dur="14.2s" repeatCount="indefinite" keyTimes="{key}" values="{ys}"/><animate attributeName="opacity" begin="3.2s" dur="14.2s" repeatCount="indefinite" keyTimes="{key}" values="0;0;1;1;1;1;1;1;0"/></circle>')
    lines.append(f'<text x="48" y="520" fill="{muted}" font-size="10" font-family="ui-monospace,Menlo,Consolas,monospace">dense portrait + sparse travelling swarm · theme={theme}</text>')
    # Info panel
    lines.append(f'<text x="505" y="93" fill="{chrome}" font-size="13" font-weight="700" font-family="ui-monospace,Menlo,Consolas,monospace">SYSTEM.INFO</text>')
    lines.append(f'<rect x="495" y="108" width="645" height="442" rx="14" fill="{panel2}" stroke="{border}"/>')
    rows=[
      ("Subject","THIRD NATURE"),("Role","Creative Developer · Full-Stack · SaaS"),("Origin","Independent digital practice"),("Education","—"),("Status","BUILDING + EXPERIMENTING + SHIPPING"),("ToolChain","VS Code · Git · Figma · Xcode · Android Studio"),
      ("Core.Lang","TypeScript · JavaScript · Python · Swift · C#"),("Core.Frontend","Next.js · React · Tailwind · SwiftUI"),("Core.Backend","Node.js · APIs · .NET"),("Core.Database","Supabase · PostgreSQL · MongoDB"),("Core.Infra","GitHub · GitHub Actions · Vercel"),
      ("Grid.Mail","wearethirdnature@gmail.com"),("Grid.Portfolio","COMING SOON"),("Grid.LinkedIn","—"),("Grid.GitHub","github.com/thirdnaturedev"),("Grid.Facebook","—")]
    y=141
    for label,val in rows:
        y+=23
        # leaders generated, not hand-tuned
        nlead=max(3, 38-len(label)-min(24,len(val)//2))
        leaders='·'*nlead
        lines.append(f'<text x="520" y="{y}" fill="{muted}" font-size="14" font-family="ui-monospace,Menlo,Consolas,monospace" textLength="575" lengthAdjust="spacingAndGlyphs">{esc(label)} {leaders} {esc(val)}</text>')
    lines.append(f'<text x="520" y="523" fill="{ACCENT}" font-size="12" font-family="ui-monospace,Menlo,Consolas,monospace">For Things That Don’t Exist Yet.</text>')
    lines.append('</svg>')
    return ''.join(lines)


def main():
    Path("data").mkdir(exist_ok=True)
    img=fetch_avatar()
    dark_bits=portrait_bits(img,True)
    light_bits=portrait_bits(img,False)
    p1=render_mark("TN")
    p2=match(p1, render_mark("</>"))
    p3=match(p2, render_mark(">_"))
    np.savez_compressed("data/profile_points.npz", dark=dark_bits, light=light_bits, logo1=p1, logo2=p2, logo3=p3)
    Path("dark.svg").write_text(svg("dark",dark_bits,(p1,p2,p3)),encoding="utf-8")
    Path("light.svg").write_text(svg("light",light_bits,(p1,p2,p3)),encoding="utf-8")
    print("generated dark.svg, light.svg and data/profile_points.npz")

if __name__=="__main__": main()

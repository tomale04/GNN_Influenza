import sys,urllib.request,urllib.parse,time
def cdx(url, to="20190501", tries=4):
    q=urllib.parse.urlencode({"url":url,"output":"txt","fl":"timestamp,original,statuscode,length","to":to,"collapse":"digest"})
    for i in range(tries):
        try:
            t=urllib.request.urlopen("https://web.archive.org/cdx/search/cdx?"+q,timeout=90).read().decode()
            if "<html" in t[:200].lower(): raise RuntimeError("html")
            return t
        except Exception as e:
            time.sleep(5*(i+1))
    return "FAILED"
for u in sys.argv[1:]:
    print("==",u); print(cdx(u)[:3000])

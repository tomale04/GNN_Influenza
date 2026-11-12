"""List all object versions in the public nextstrain-data S3 bucket under a prefix."""
import sys, urllib.request, urllib.parse, xml.etree.ElementTree as ET, json
NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
BASE = "https://nextstrain-data.s3.amazonaws.com/"

def list_versions(prefix):
    out, km, vm = [], None, None
    while True:
        q = {"versions": "", "prefix": prefix}
        if km: q["key-marker"] = km
        if vm: q["version-id-marker"] = vm
        url = BASE + "?" + urllib.parse.urlencode(q).replace("versions=", "versions")
        root = ET.fromstring(urllib.request.urlopen(url, timeout=60).read())
        for tag in ("Version", "DeleteMarker"):
            for v in root.findall(f"s:{tag}", NS):
                out.append({"key": v.find("s:Key", NS).text,
                            "version_id": v.find("s:VersionId", NS).text,
                            "last_modified": v.find("s:LastModified", NS).text,
                            "size": int(v.find("s:Size", NS).text) if v.find("s:Size", NS) is not None else None,
                            "delete_marker": tag == "DeleteMarker"})
        if root.find("s:IsTruncated", NS).text != "true": break
        km = root.find("s:NextKeyMarker", NS).text
        vm = root.find("s:NextVersionIdMarker", NS).text
    return out

if __name__ == "__main__":
    print(json.dumps(list_versions(sys.argv[1])))

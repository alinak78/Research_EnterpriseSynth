"""Copy compiled PDF, clear identifying metadata, and run manuscript preflight."""
from pathlib import Path
import json,re
from pypdf import PdfReader,PdfWriter
R=Path(__file__).resolve().parents[1]
def main():
 src=R/'paper/build/main.pdf';dest=R/'output/pdf/IntelliSys2027_EnterpriseSynth_Draft.pdf'
 reader=PdfReader(src);writer=PdfWriter();writer.clone_document_from_reader(reader)
 writer.metadata=None;writer.add_metadata({'/Title':'Validity Is Not Utility: A Paired Audit of Constraint Repair and Class Coverage in Synthetic Tabular Data','/Author':'','/Subject':'Anonymous IntelliSys 2027 research draft'})
 if '/Metadata' in writer.root_object:del writer.root_object['/Metadata']
 dest.parent.mkdir(parents=True,exist_ok=True)
 with dest.open('wb') as f:writer.write(f)
 reader=PdfReader(dest);text='\n'.join(p.extract_text() for p in reader.pages)
 forbidden=['alina','rashmi','thimmaraju','anote','alinak78','/Users/','@gmail','github.com/alinak78','github.com/anote-ai']
 assert not any(s.lower() in (text+str(reader.metadata)).lower() for s in forbidden)
 assert len(reader.pages)<=18 and dest.stat().st_size<20_000_000
 assert not reader.attachments
 assert '/Metadata' not in reader.trailer['/Root']
 assert not reader.metadata.get('/Author')
 assert not re.search(r'\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b',text)
 source=(R/'paper/main.tex').read_text();abstract=source.split('\\begin{abstract}')[1].split('\\keywords')[0];words=len(abstract.split());assert 150<=words<=250
 log=(R/'paper/build/main.log').read_text();assert 'Overfull' not in log and 'undefined' not in log
 fonts=[]
 visited=set()
 def inspect_fonts(resources):
  resources=resources.get_object()
  for font in (resources['/Font'].get_object() if '/Font' in resources else {}).values():
   font=font.get_object();desc=font.get('/FontDescriptor')
   if desc is None:
    descendants=font.get('/DescendantFonts',[])
    if descendants:desc=descendants[0].get_object().get('/FontDescriptor')
   assert desc is not None,str(font)
   desc=desc.get_object();assert any(k in desc for k in ['/FontFile','/FontFile2','/FontFile3']),str(font)
   fonts.append(str(font.get('/BaseFont')))
  for obj in (resources['/XObject'].get_object() if '/XObject' in resources else {}).values():
   obj=obj.get_object()
   if id(obj) in visited:continue
   visited.add(id(obj))
   if '/Resources' in obj:inspect_fonts(obj['/Resources'])
 for page in reader.pages:inspect_fonts(page['/Resources'])
 reference_start=next(i+1 for i,p in enumerate(reader.pages) if re.search(r'(?m)^References$',p.extract_text()))
 report=dict(pages=len(reader.pages),references_start_page=reference_start,abstract_words=words,bytes=dest.stat().st_size,metadata=dict(reader.metadata),embedded_fonts=sorted(set(fonts)),known_identity_checks='pass',latex_overfull_and_unresolved='none',status='human scientific review and GenAI-policy assessment outstanding')
 (R/'results/pdf_preflight.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()

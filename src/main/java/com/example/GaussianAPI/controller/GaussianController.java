package com.example.GaussianAPI.controller;

import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.multipart.commons.CommonsMultipartResolver;

import io.swagger.annotations.Api;
import io.swagger.annotations.ApiImplicitParam;
import io.swagger.annotations.ApiImplicitParams;
import io.swagger.annotations.ApiOperation;
import io.swagger.annotations.ApiParam;


import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.UUID;
import java.util.stream.Collectors;

import javax.servlet.http.HttpServletRequest;
import java.io.File;
import java.io.IOException;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Bean;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.core.io.Resource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.util.StringUtils;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.RestController;
import org.apache.commons.io.FileUtils;
@RestController
public class GaussianController {

	/* private static final Logger logger = LoggerFactory.getLogger(GaussianController.class);

    @Autowired
    private FileStorageService fileStorageService;

    @PostMapping("/uploadFile")
    public UploadFileResponse uploadFile1(@RequestParam("file") MultipartFile file) {
        String fileName = fileStorageService.storeFile(file);

        String fileDownloadUri = ServletUriComponentsBuilder.fromCurrentContextPath()
                .path("/downloadFile/")
                .path(fileName)
                .toUriString();




        return new UploadFileResponse(fileName, fileDownloadUri,
        		file.getContentType(), file.getSize());
    }

    @PostMapping("/uploadMultipleFiles")
    public List<UploadFileResponse> uploadMultipleFiles(@RequestParam("files") MultipartFile[] files) {
        return Arrays.asList(files)
                .stream()
                .map(file -> uploadFile1(file))
                .collect(Collectors.toList());
    }

    @GetMapping("/downloadFile/{fileName:.+}")
    public ResponseEntity<Resource> downloadFile(@PathVariable String fileName, HttpServletRequest request) {
        // Load file as Resource
        Resource resource = fileStorageService.loadFileAsResource(fileName);

        // Try to determine file's content type
        String contentType = null;
        try {
            contentType = request.getServletContext().getMimeType(resource.getFile().getAbsolutePath());
        } catch (IOException ex) {
            logger.info("Could not determine file type.");
        }

        // Fallback to the default content type if type could not be determined
        if(contentType == null) {
            contentType = "application/octet-stream";
        }

        return ResponseEntity.ok()
                .contentType(MediaType.parseMediaType(contentType))
                .header(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=\"" + resource.getFilename() + "\"")
                .body(resource);
    }
	 */
	@PostMapping(value = "/GPrep", consumes = {MediaType.MULTIPART_FORM_DATA_VALUE})
	@ApiOperation(value = "Make a POST request to upload the file",
	produces = "application/json", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
	@ApiImplicitParams({
		@ApiImplicitParam(name = "file", value="File to be analyzed", dataType = "file", paramType = "form", required = true)})
	public ResponseEntity<String> uploadFile(
			@ApiParam(name = "file", required = true)
			@RequestPart(value = "file", required = true) MultipartFile file,
			@RequestParam(value = "sigma", defaultValue = "0.05") Double sigma,
			@RequestParam(value = "beta", defaultValue = "3.0") Double beta,
			@RequestParam(value = "theta", defaultValue = "1.0") Double theta,
			@RequestParam(value = "delta", defaultValue = "1.0") Double delta,
			@RequestParam(value = "d", defaultValue = "1.0") Double d,
			@RequestParam(value = "c", defaultValue = "5.0") Double c,
			@RequestParam(value = "N", defaultValue = "100") Integer N,
			@RequestParam(value = "b2dropUsername", defaultValue = "username") String b2dropusername,
			@RequestParam(value = "b2dropPassword", defaultValue = "password") String b2droppassword) throws IOException
			//@RequestParam(value = "email", defaultValue = "gaussianrestapi@gmail.com") String email) throws IOException
	{
		String fname="test"+UUID.randomUUID().toString();
		File testFile = new File("/opt/gaussian_data/"+fname);
		
	
		try {
			
			FileUtils.writeByteArrayToFile(testFile, file.getBytes());
			List<String> lines = FileUtils.readLines(testFile);
			lines.forEach(line -> System.out.println(line));
			Runtime.getRuntime().exec("chmod 777 /opt/gaussian_data/"+fname);
		} catch (IOException e) {
			e.printStackTrace();
			return new ResponseEntity<String>("Failed", HttpStatus.INTERNAL_SERVER_ERROR);
		}
		
		String pythonScriptPath=this.getClass().getClassLoader().getResource("scripts/GP.py").getPath();
		String res="";


		String[] cmd = new String[12];
		cmd[0] = "python"; // check version of installed python: python -V
		cmd[1] = pythonScriptPath;
		cmd[2] = testFile.getAbsolutePath();
		cmd[3] = sigma.toString();
		cmd[4] = beta.toString();
		cmd[5] = theta.toString();
		cmd[6] = delta.toString();
		cmd[7] = d.toString();
		cmd[8] = c.toString();
		cmd[9] = N.toString();
		cmd[10] = b2dropusername;
		cmd[11] = b2droppassword;


		// create runtime to execute external command
		Runtime rt = Runtime.getRuntime();
		Process pr;



		try {
			pr = rt.exec(cmd);
			// TODO Auto-generated catch block

			// retrieve output from python script
			BufferedReader bfr = new BufferedReader(new InputStreamReader(pr.getInputStream()));
			String line = "";


			while((line = bfr.readLine()) != null) {
				// display each output line form python script
				//System.out.println(line);
				//res=res+"<a href=\""+line+"\" target=\"_blank\">"+line+"</a>"+"<br>";

				//String url = "<html><a href=" + line + ">" + line + "<//a><//html>.";
				//res=res+url+"<br>";
				//String link = "<a href=\"http://google.com\" target=\"_blank\">http://google.com</a>";
               res=res+line+"\n";

			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		return ResponseEntity.status(HttpStatus.OK)
		        .body(res);
	}



	//http://194.149.135.58:8080/GaussianAPI/GPrep?email=bojana.koteska@gmail.com&file=http://194.149.135.58:8080/GaussianAPI/downloadFile/reference_dataB.xyz&sigma=0.05&beta_global=3.0&theta_global=1.0&cutoff_CC=2.0&start_CC=1.1&cutoff_CH=1.7&start_CH=0.8&cutoff_CO=2.0&start_CO=1.1&cutoff_OO=2.0&start_OO=1.1&cutoff_OH=1.7&start_OH=0.8&cutoff_HH=1.2&start_HH=0.4&c=5.0&N=100;
	// python GP.py http://194.149.135.58:8080/GaussianAPI/downloadFile/reference_dataB.xyz 0.05 3.0 1.0 2.0 1.1 1.7 0.8 2.0 1.1 2.0 1.1 1.7 0.8 1.2 0.4 5.0 100
	// python GPurl.py https://gaussian.chem-api.finki.ukim.mk/static/reference_data.xyz 0.05 3.0 1.0 1.0 1.0 5.0 100 bojana.koteska@gmail.com
	//python get_distances.py -c 5.0 -o tmp.rep reference_data.xyz


	@GetMapping("/GPrepRemote")
	public String GPrepRemote(
			@RequestParam(value = "file", defaultValue = "https://gaussian.chem-api.finki.ukim.mk/static/reference_data.xyz") String file,
			@RequestParam(value = "sigma", defaultValue = "0.05") Double sigma,
			@RequestParam(value = "beta", defaultValue = "3.0") Double beta,
			@RequestParam(value = "theta", defaultValue = "1.0") Double theta,
			@RequestParam(value = "delta", defaultValue = "1.0") Double delta,
			@RequestParam(value = "d", defaultValue = "1.0") Double d,
			@RequestParam(value = "c", defaultValue = "5.0") Double c,
			@RequestParam(value = "N", defaultValue = "100") Integer N,
			@RequestParam(value = "b2dropUsername", defaultValue = "username") String b2dropusername,
			@RequestParam(value = "b2dropPassword", defaultValue = "password") String b2droppassword) throws IOException
			//@RequestParam(value = "email", defaultValue = "gaussianrestapi@gmail.com") String email) throws IOException

	{


		String pythonScriptPath=this.getClass().getClassLoader().getResource("scripts/GPurl.py").getPath();
		String res="";


		String[] cmd = new String[12];
		cmd[0] = "python"; // check version of installed python: python -V
		cmd[1] = pythonScriptPath;
		cmd[2] = file;
		cmd[3] = sigma.toString();
		cmd[4] = beta.toString();
		cmd[5] = theta.toString();
		cmd[6] = delta.toString();
		cmd[7] = d.toString();
		cmd[8] = c.toString();
		cmd[9] = N.toString();
		cmd[10] = b2dropusername;
		cmd[11] = b2droppassword;





		// create runtime to execute external command
		Runtime rt = Runtime.getRuntime();
		Process pr;



		try {
			pr = rt.exec(cmd);
			// TODO Auto-generated catch block

			// retrieve output from python script
			BufferedReader bfr = new BufferedReader(new InputStreamReader(pr.getInputStream()));
			String line = "";


			while((line = bfr.readLine()) != null) {
				// display each output line form python script
				//System.out.println(line);
				//res=res+"<a href=\""+line+"\" target=\"_blank\">"+line+"</a>"+"<br>";

				//String url = "<html><a href=" + line + ">" + line + "<//a><//html>.";
				//res=res+url+"<br>";
				//String link = "<a href=\"http://google.com\" target=\"_blank\">http://google.com</a>";
				res=res+line+"\n";

			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		//res= System.getProperty("catalina.base") + File.separator + "bin";

		//MockMultipartFile multipartFile = new MockMultipartFile("file", "hello.txt", MediaType.TEXT_PLAIN_VALUE, "Hello, World!".getBytes());


		//MultipartFile multipartFile = new MockMultipartFile("C-C_GPrep_SI.skf", new FileInputStream(new File(res+"/C-C_GPrep_SI.skf")));



		/*Resource invoicesResource = multipartFile.getResource();

		RestTemplate restTemplate=new RestTemplate();
	    LinkedMultiValueMap<String, Object> parts = new LinkedMultiValueMap<>();
	    parts.add("file", multipartFile);

	    HttpHeaders httpHeaders = new HttpHeaders();
	    httpHeaders.setContentType(MediaType.MULTIPART_FORM_DATA);

	    HttpEntity<LinkedMultiValueMap<String, Object>> httpEntity = new HttpEntity<>(parts, httpHeaders);

	    restTemplate.postForEntity("http://194.149.135.58:8080/GaussianAPI/uploadFile/", httpEntity, String.class);

		 */
		return res;

	}
}

